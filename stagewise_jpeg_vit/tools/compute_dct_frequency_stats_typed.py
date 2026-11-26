#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dct_manip as dm
from utils.custom_transforms import DCDiffQuantDCT


COMPONENTS = ("y", "cb", "cr")
TYPES = ("dc_start", "dc_delta", "ac")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute typed quantized DCT statistics after standard DC DPCM."
    )
    parser.add_argument("--index", default="assets/indexbase_train.csv")
    parser.add_argument("--data-root", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--output-prefix", default="outputs/dct_stats/imagenet1k_train_dct_freq_stats_dcdiff_typed")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="standard", choices=("standard",))
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args()


def init_worker(mode):
    torch.set_num_threads(1)
    global DCDIFF
    DCDIFF = DCDiffQuantDCT(mode=mode)


def empty_stats():
    stats = {}
    for component in COMPONENTS:
        for value_type in TYPES:
            stats[f"count_{component}_{value_type}"] = 0
            stats[f"sum_{component}_{value_type}"] = torch.zeros(8, 8, dtype=torch.float64)
            stats[f"sumsq_{component}_{value_type}"] = torch.zeros(8, 8, dtype=torch.float64)
    return stats


def _accumulate_values(stats, component, value_type, values):
    if values.numel() == 0:
        return
    values = values.to(torch.float64)
    stats[f"count_{component}_{value_type}"] += int(values.shape[0])
    stats[f"sum_{component}_{value_type}"] += values.sum(dim=0)
    stats[f"sumsq_{component}_{value_type}"] += (values * values).sum(dim=0)


def _split_typed_values(component_tensor, component_name):
    component_blocks = component_tensor.reshape(-1, 8, 8)
    if component_name == "y":
        _, height, width, _, _ = component_tensor.shape
        even_height = height - (height % 2)
        even_width = width - (width % 2)
        typed = component_tensor[:, :even_height, :even_width]
        dc_values = typed[:, :, :, 0, 0].reshape(1, even_height // 2, 2, even_width // 2, 2)
        dc_values = dc_values.permute(0, 1, 3, 2, 4).reshape(-1)
    else:
        dc_values = component_tensor[:, :, 0, 0].reshape(-1)
    dc_start = component_tensor.new_empty((1, 8, 8))
    dc_start[0] = 0
    dc_start[0, 0, 0] = dc_values[0]
    dc_delta = component_tensor.new_empty((max(dc_values.numel() - 1, 0), 8, 8))
    if dc_delta.numel() > 0:
        dc_delta.zero_()
        dc_delta[:, 0, 0] = dc_values[1:]
    ac = component_blocks.clone()
    ac[:, 0, 0] = 0
    return dc_start, dc_delta, ac


def accumulate_one(stats, root, relpath):
    _, _, y_coeff, cbcr_coeff = dm.read_coefficients(str(Path(root) / relpath))
    if cbcr_coeff is None:
        y_diff = DCDIFF(y_coeff)
        cbcr_diff = None
    else:
        y_diff, cbcr_diff = DCDIFF((y_coeff, cbcr_coeff))

    dc_start, dc_delta, ac = _split_typed_values(y_diff, "y")
    _accumulate_values(stats, "y", "dc_start", dc_start)
    _accumulate_values(stats, "y", "dc_delta", dc_delta)
    _accumulate_values(stats, "y", "ac", ac)

    if cbcr_diff is not None:
        dc_start, dc_delta, ac = _split_typed_values(cbcr_diff[0], "cb")
        _accumulate_values(stats, "cb", "dc_start", dc_start)
        _accumulate_values(stats, "cb", "dc_delta", dc_delta)
        _accumulate_values(stats, "cb", "ac", ac)

        dc_start, dc_delta, ac = _split_typed_values(cbcr_diff[1], "cr")
        _accumulate_values(stats, "cr", "dc_start", dc_start)
        _accumulate_values(stats, "cr", "dc_delta", dc_delta)
        _accumulate_values(stats, "cr", "ac", ac)

    return stats


def worker_chunk(task):
    root, relpaths = task
    stats = empty_stats()
    for relpath in relpaths:
        accumulate_one(stats, root, relpath)
    return stats, len(relpaths)


def merge_stats(dst, src):
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            dst[key] += value
        else:
            dst[key] += int(value)


def finalize(stats, component, value_type):
    count = stats[f"count_{component}_{value_type}"]
    if count <= 0:
        mean = torch.zeros(8, 8, dtype=torch.float64)
        std = torch.zeros(8, 8, dtype=torch.float64)
        return mean, std
    total = stats[f"sum_{component}_{value_type}"]
    total_sq = stats[f"sumsq_{component}_{value_type}"]
    mean = total / count
    variance = (total_sq / count) - (mean * mean)
    return mean, variance.clamp_min(0.0).sqrt()


def tensor_to_list(tensor):
    return [[float(x) for x in row] for row in tensor.tolist()]


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)

    index = Path(args.index)
    data_root = Path(args.data_root)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(index).iloc[:, 0].astype(str).tolist()
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    path_chunks = [paths[start : start + args.chunksize] for start in range(0, len(paths), args.chunksize)]
    tasks = [(str(data_root), chunk) for chunk in path_chunks]
    workers = max(1, min(args.workers, os.cpu_count() or args.workers))

    print(f"host={os.uname().nodename}", flush=True)
    print(f"index={index.resolve()}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"images={len(paths)}", flush=True)
    print(f"chunks={len(tasks)}", flush=True)
    print(f"workers={workers} chunksize={args.chunksize}", flush=True)
    print(f"mode={args.mode}", flush=True)

    stats = empty_stats()
    start = time.time()
    processed = 0

    if workers == 1:
        init_worker(args.mode)
        iterator = map(worker_chunk, tasks)
        for partial, num_images in iterator:
            merge_stats(stats, partial)
            processed += num_images
            if processed % args.progress_every < num_images or processed == len(paths):
                elapsed = time.time() - start
                print(f"processed={processed}/{len(paths)} elapsed={elapsed/60:.2f}min img/s={processed/elapsed:.2f}", flush=True)
    else:
        with mp.Pool(workers, initializer=init_worker, initargs=(args.mode,)) as pool:
            iterator = pool.imap_unordered(worker_chunk, tasks, chunksize=1)
            for partial, num_images in iterator:
                merge_stats(stats, partial)
                processed += num_images
                if processed % args.progress_every < num_images or processed == len(paths):
                    elapsed = time.time() - start
                    print(f"processed={processed}/{len(paths)} elapsed={elapsed/60:.2f}min img/s={processed/elapsed:.2f}", flush=True)

    elapsed = time.time() - start
    result = {
        "meta": {
            "index": str(index.resolve()),
            "data_root": str(data_root),
            "images": len(paths),
            "workers": workers,
            "chunksize": args.chunksize,
            "mode": args.mode,
            "elapsed_sec": elapsed,
        },
        "raw": stats,
    }

    json_result = {"meta": result["meta"]}
    for component in COMPONENTS:
        for value_type in TYPES:
            mean, std = finalize(stats, component, value_type)
            result[f"mean_q_{component}_{value_type}"] = mean.float()
            result[f"std_q_{component}_{value_type}"] = std.float()
            result[f"count_q_{component}_{value_type}"] = stats[f"count_{component}_{value_type}"]
            key = f"Q_{component.upper()}_{value_type.upper()}"
            json_result[f"mean_{key}"] = tensor_to_list(mean)
            json_result[f"std_{key}"] = tensor_to_list(std)
            json_result[f"count_{key}"] = stats[f"count_{component}_{value_type}"]

    pt_path = output_prefix.with_suffix(".pt")
    json_path = output_prefix.with_suffix(".json")
    torch.save(result, pt_path)
    json_path.write_text(json.dumps(json_result, indent=2), encoding="utf-8")

    print(f"elapsed={elapsed/60:.2f}min img/s={len(paths)/elapsed:.2f}", flush=True)
    print(f"saved_pt={pt_path}", flush=True)
    print(f"saved_json={json_path}", flush=True)
    for component in COMPONENTS:
        for value_type in TYPES:
            std = result[f"std_q_{component}_{value_type}"]
            count = result[f"count_q_{component}_{value_type}"]
            print(
                f"std_Q_{component}_{value_type}: count={count} dc={std[0,0].item():.6f} "
                f"min={std.min().item():.6f} max={std.max().item():.6f} mean={std.mean().item():.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
