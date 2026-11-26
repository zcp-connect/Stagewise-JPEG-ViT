#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import datasets as ds


def percentile_dict(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    quantiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    out = {f"{prefix}_p{str(q).replace('.', '_')}": float(np.percentile(values, q)) for q in quantiles}
    out[f"{prefix}_mean"] = float(values.mean())
    out[f"{prefix}_std"] = float(values.std())
    out[f"{prefix}_count"] = int(values.size)
    return out


def length_hist(values, bins):
    values = np.asarray(values, dtype=np.int64)
    hist = {}
    for lo, hi in bins:
        if hi is None:
            key = f">={lo}"
            count = int((values >= lo).sum())
        else:
            key = f"[{lo},{hi})"
            count = int(((values >= lo) & (values < hi)).sum())
        hist[key] = count
    return hist


def sample_indices(n, limit, seed):
    if limit is None or limit <= 0 or limit >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def resolve_path(basepath, relpath):
    if basepath:
        return os.path.join(basepath, relpath)
    return relpath


def _block_event_counts(block, include_eob):
    rlc_count = 1
    huff_count = 1
    zero_run = 0
    for coeff in block[1:]:
        coeff = int(coeff)
        if coeff == 0:
            zero_run += 1
            continue
        while zero_run >= 16:
            rlc_count += 1
            huff_count += 1
            zero_run -= 16
        rlc_count += 1
        huff_count += 1
        zero_run = 0
    if include_eob and zero_run > 0:
        huff_count += 1
    return rlc_count, huff_count


def count_true_events(blocks_zz):
    if blocks_zz.dim() != 3 or tuple(blocks_zz.shape) != (196, 6, 64):
        raise ValueError(f"expected blocks_zz shape (196, 6, 64), got {tuple(blocks_zz.shape)}")
    if blocks_zz.is_floating_point():
        blocks_zz = blocks_zz.round().to(torch.long)
    blocks_list = blocks_zz.tolist()
    rlc_counts = []
    huff_counts = []
    for mcu_blocks in blocks_list:
        rlc_total = 0
        huff_total = 0
        for block in mcu_blocks:
            rlc_count, huff_count = _block_event_counts(block, include_eob=True)
            rlc_total += rlc_count
            huff_total += huff_count
        rlc_counts.append(rlc_total)
        huff_counts.append(huff_total)
    return np.asarray(rlc_counts, dtype=np.int32), np.asarray(huff_counts, dtype=np.int32)


def collect_one(imgpath, split, dtype, use_randaugment, ops_list, num_ops, ops_magnitude):
    stage_for_loader = "quant_dct" if use_randaugment else None
    _, _quant, y_out, cbcr_out = ds.load_augmented_mcu_coefficients(
        imgpath,
        type=split,
        dtype_resize=dtype,
        stage=stage_for_loader,
        ops_list=ops_list,
        num_ops=num_ops,
        ops_magnitude=ops_magnitude,
    )
    blocks_zz = ds._zigzag_blocks_for_rlc(y_out, cbcr_out)
    return count_true_events(blocks_zz)


def summarize_lengths(values, lmax, prefix):
    values = np.asarray(values, dtype=np.int32)
    result = percentile_dict(values, prefix)
    result[f"{prefix}_hist"] = length_hist(
        values,
        [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192), (192, 224), (224, 256), (256, 288), (288, 320), (320, 384), (384, None)],
    )
    result[f"{prefix}_truncated_mcus_at_lmax"] = int((values > lmax).sum())
    result[f"{prefix}_truncated_ratio_at_lmax"] = float((values > lmax).mean())
    for cutoff in [64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]:
        result[f"{prefix}_gt_{cutoff}"] = int((values > cutoff).sum())
        result[f"{prefix}_gt_{cutoff}_ratio"] = float((values > cutoff).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description="Count true untruncated per-MCU RLC/Huffman event budgets.")
    parser.add_argument("--index-path", default=str(REPO_ROOT / "assets" / "indexbase_train.csv"))
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=320)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--ops-list", default="")
    parser.add_argument("--num-ops", type=int, default=2)
    parser.add_argument("--ops-magnitude", type=int, default=3)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "phase_a" / "event_budget_count_only.json"))
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    index = pd.read_csv(args.index_path).to_numpy()
    chosen = sample_indices(len(index), args.limit, args.seed)
    ops_list = None if args.ops_list == "" else args.ops_list.split(",")

    rlc_lengths = []
    huff_lengths = []
    image_count = 0
    error_count = 0
    t0 = time.perf_counter()

    for j, idx in enumerate(chosen, start=1):
        relpath = index[idx, 0]
        imgpath = resolve_path(args.basepath, relpath)
        try:
            rlc_event_len, huff_event_len = collect_one(
                imgpath,
                args.split,
                dtype,
                args.randaugment,
                ops_list,
                args.num_ops,
                args.ops_magnitude,
            )
        except Exception as exc:
            error_count += 1
            if error_count <= 10:
                print(f"[error] {imgpath}: {type(exc).__name__}: {exc}", flush=True)
            continue
        image_count += 1
        rlc_lengths.append(rlc_event_len)
        huff_lengths.append(huff_event_len)
        if args.progress_every > 0 and j % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"processed={j}/{len(chosen)} ok_images={image_count} elapsed={elapsed:.1f}s imgs/s={image_count/max(elapsed,1e-9):.2f}", flush=True)

    if not rlc_lengths:
        raise RuntimeError("No samples processed successfully")

    rlc_lengths = np.concatenate(rlc_lengths)
    huff_lengths = np.concatenate(huff_lengths)
    result = {
        "index_path": args.index_path,
        "basepath": args.basepath,
        "split": args.split,
        "limit_requested": args.limit,
        "images_seen": int(len(chosen)),
        "images_ok": int(image_count),
        "errors": int(error_count),
        "mcus": int(rlc_lengths.size),
        "lmax": int(args.lmax),
        "randaugment": bool(args.randaugment),
        "num_ops": int(args.num_ops),
        "ops_magnitude": int(args.ops_magnitude),
        "elapsed_sec": float(time.perf_counter() - t0),
        "note": "Counts are true untruncated event counts. Truncation ratio uses true_count > lmax.",
    }
    result.update(summarize_lengths(rlc_lengths, args.lmax, "rlc_event_len"))
    result.update(summarize_lengths(huff_lengths, args.lmax, "huff_event_len"))
    result["huff_minus_rlc_mean"] = float((huff_lengths - rlc_lengths).mean())
    result["huff_minus_rlc_p99"] = float(np.percentile(huff_lengths - rlc_lengths, 99))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
