#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd
import torch

import dct_manip as dm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ImageNet DCT frequency-wise mean/std statistics."
    )
    parser.add_argument("--index", default="assets/indexbase_train.csv")
    parser.add_argument("--data-root", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--output-prefix", default="outputs/dct_stats/imagenet1k_train_dct_freq_stats")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clamp-min", type=float, default=-1024.0)
    parser.add_argument("--clamp-max", type=float, default=1016.0)
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args()


def init_worker(clamp_min, clamp_max):
    torch.set_num_threads(1)
    global CLAMP_MIN
    global CLAMP_MAX
    CLAMP_MIN = clamp_min
    CLAMP_MAX = clamp_max


def empty_stats():
    return {
        "count_y": 0,
        "count_cb": 0,
        "count_cr": 0,
        "sum_q_y": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_q_y": torch.zeros(8, 8, dtype=torch.float64),
        "sum_q_cb": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_q_cb": torch.zeros(8, 8, dtype=torch.float64),
        "sum_q_cr": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_q_cr": torch.zeros(8, 8, dtype=torch.float64),
        "sum_f_y": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_f_y": torch.zeros(8, 8, dtype=torch.float64),
        "sum_f_cb": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_f_cb": torch.zeros(8, 8, dtype=torch.float64),
        "sum_f_cr": torch.zeros(8, 8, dtype=torch.float64),
        "sumsq_f_cr": torch.zeros(8, 8, dtype=torch.float64),
    }


def accumulate_one(stats, root, relpath):
    _, quant_table, y_coeff, cbcr_coeff = dm.read_coefficients(str(Path(root) / relpath))

    y_quant = y_coeff.to(torch.float64)
    stats["count_y"] += int(y_coeff.shape[0] * y_coeff.shape[1] * y_coeff.shape[2])
    stats["sum_q_y"] += y_quant.sum(dim=(0, 1, 2))
    stats["sumsq_q_y"] += (y_quant * y_quant).sum(dim=(0, 1, 2))

    y_float = (y_quant * quant_table[0].to(torch.float64)).clamp(CLAMP_MIN, CLAMP_MAX)
    stats["sum_f_y"] += y_float.sum(dim=(0, 1, 2))
    stats["sumsq_f_y"] += (y_float * y_float).sum(dim=(0, 1, 2))

    if cbcr_coeff is not None:
        cbcr_quant = cbcr_coeff.to(torch.float64)
        chroma_count = int(cbcr_coeff.shape[1] * cbcr_coeff.shape[2])

        cb_quant = cbcr_quant[0]
        cr_quant = cbcr_quant[1]
        stats["count_cb"] += chroma_count
        stats["count_cr"] += chroma_count
        stats["sum_q_cb"] += cb_quant.sum(dim=(0, 1))
        stats["sumsq_q_cb"] += (cb_quant * cb_quant).sum(dim=(0, 1))
        stats["sum_q_cr"] += cr_quant.sum(dim=(0, 1))
        stats["sumsq_q_cr"] += (cr_quant * cr_quant).sum(dim=(0, 1))

        cb_float = (cb_quant * quant_table[1].to(torch.float64)).clamp(CLAMP_MIN, CLAMP_MAX)
        cr_float = (cr_quant * quant_table[2].to(torch.float64)).clamp(CLAMP_MIN, CLAMP_MAX)
        stats["sum_f_cb"] += cb_float.sum(dim=(0, 1))
        stats["sumsq_f_cb"] += (cb_float * cb_float).sum(dim=(0, 1))
        stats["sum_f_cr"] += cr_float.sum(dim=(0, 1))
        stats["sumsq_f_cr"] += (cr_float * cr_float).sum(dim=(0, 1))

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


def finalize_component(stats, prefix, component):
    count = stats[f"count_{component}"]
    if count <= 0:
        raise ValueError(f"No samples accumulated for component {component}")

    sum_key = f"sum_{prefix}_{component}"
    sumsq_key = f"sumsq_{prefix}_{component}"
    mean = stats[sum_key] / count
    variance = (stats[sumsq_key] / count) - (mean * mean)
    variance = variance.clamp_min(0.0)
    std = variance.sqrt()
    return mean, std


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

    path_chunks = [
        paths[start : start + args.chunksize]
        for start in range(0, len(paths), args.chunksize)
    ]
    tasks = [(str(data_root), chunk) for chunk in path_chunks]
    total = len(tasks)
    workers = max(1, min(args.workers, os.cpu_count() or args.workers))

    print(f"host={os.uname().nodename}", flush=True)
    print(f"index={index.resolve()}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"images={len(paths)}", flush=True)
    print(f"chunks={total}", flush=True)
    print(f"workers={workers} chunksize={args.chunksize}", flush=True)
    print(f"clamp=[{args.clamp_min}, {args.clamp_max}]", flush=True)

    stats = empty_stats()
    start = time.time()

    if workers == 1:
        init_worker(args.clamp_min, args.clamp_max)
        iterator = map(worker_chunk, tasks)
        processed = 0
        for partial, num_images in iterator:
            merge_stats(stats, partial)
            processed += num_images
            if processed % args.progress_every < num_images or processed == len(paths):
                elapsed = time.time() - start
                print(
                    f"processed={processed}/{len(paths)} elapsed={elapsed/60:.2f}min "
                    f"img/s={processed/elapsed:.2f}",
                    flush=True,
                )
    else:
        with mp.Pool(
            workers,
            initializer=init_worker,
            initargs=(args.clamp_min, args.clamp_max),
        ) as pool:
            iterator = pool.imap_unordered(worker_chunk, tasks, chunksize=1)
            processed = 0
            for partial, num_images in iterator:
                merge_stats(stats, partial)
                processed += num_images
                if processed % args.progress_every < num_images or processed == len(paths):
                    elapsed = time.time() - start
                    print(
                        f"processed={processed}/{len(paths)} elapsed={elapsed/60:.2f}min "
                        f"img/s={processed/elapsed:.2f}",
                        flush=True,
                    )

    elapsed = time.time() - start

    result = {
        "meta": {
            "index": str(index.resolve()),
            "data_root": str(data_root),
            "images": len(paths),
            "workers": workers,
            "chunksize": args.chunksize,
            "clamp_min": args.clamp_min,
            "clamp_max": args.clamp_max,
            "elapsed_sec": elapsed,
            "count_y": stats["count_y"],
            "count_cb": stats["count_cb"],
            "count_cr": stats["count_cr"],
        },
        "raw": stats,
    }

    for prefix in ("q", "f"):
        for component in ("y", "cb", "cr"):
            mean, std = finalize_component(stats, prefix, component)
            result[f"mean_{prefix}_{component}"] = mean
            result[f"std_{prefix}_{component}"] = std

    pt_path = output_prefix.with_suffix(".pt")
    json_path = output_prefix.with_suffix(".json")
    torch.save(result, pt_path)

    json_result = {
        "meta": result["meta"],
        "mean_Q_Y": tensor_to_list(result["mean_q_y"]),
        "std_Q_Y": tensor_to_list(result["std_q_y"]),
        "mean_Q_Cb": tensor_to_list(result["mean_q_cb"]),
        "std_Q_Cb": tensor_to_list(result["std_q_cb"]),
        "mean_Q_Cr": tensor_to_list(result["mean_q_cr"]),
        "std_Q_Cr": tensor_to_list(result["std_q_cr"]),
        "mean_F_Y": tensor_to_list(result["mean_f_y"]),
        "std_F_Y": tensor_to_list(result["std_f_y"]),
        "mean_F_Cb": tensor_to_list(result["mean_f_cb"]),
        "std_F_Cb": tensor_to_list(result["std_f_cb"]),
        "mean_F_Cr": tensor_to_list(result["mean_f_cr"]),
        "std_F_Cr": tensor_to_list(result["std_f_cr"]),
    }
    json_path.write_text(json.dumps(json_result, indent=2), encoding="utf-8")

    print(f"elapsed={elapsed/60:.2f}min img/s={len(paths)/elapsed:.2f}", flush=True)
    print(f"saved_pt={pt_path}", flush=True)
    print(f"saved_json={json_path}", flush=True)
    for prefix, display_prefix in (("q", "Q"), ("f", "F")):
        for component, display_component in (("y", "Y"), ("cb", "Cb"), ("cr", "Cr")):
            std = result[f"std_{prefix}_{component}"]
            print(
                f"std_{display_prefix}_{display_component}: "
                f"dc={std[0,0].item():.6f} min={std.min().item():.6f} "
                f"max={std.max().item():.6f} mean={std.mean().item():.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
