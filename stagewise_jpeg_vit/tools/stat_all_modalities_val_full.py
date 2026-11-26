#!/usr/bin/env bash
# tools/stat_all_modalities_val_full.py
"""
Stat true untruncated per-MCU RLC, Parsed Huffman, and Raw Byte lengths
across the full ImageNet validation set with multiprocessing.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


def _block_event_counts(block, include_eob=True):
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


def process_single_image(imgpath, split="val"):
    try:
        _, quant, y_out, cbcr_out = ds.load_augmented_mcu_coefficients(
            imgpath,
            type=split,
            dtype_resize=torch.float32,
            stage=None,
        )
        blocks_zz = ds._zigzag_blocks_for_rlc(y_out, cbcr_out)
        rlc_counts, huff_counts = count_true_events(blocks_zz)

        new_dim = torch.tensor([[224, 224]], dtype=torch.int32)
        jpeg_buffer = ds.dm.write_coefficients_to_buffer(new_dim, quant, y_out, cbcr_out)
        records = ds.bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)["mcus"]
        byte_counts = np.asarray([int(m["byte_len"]) for m in records], dtype=np.int32)

        return rlc_counts, huff_counts, byte_counts
    except Exception as exc:
        return None


def process_chunk(imgpaths, split="val"):
    results_rlc = []
    results_huff = []
    results_byte = []
    ok_count = 0
    err_count = 0
    for path in imgpaths:
        res = process_single_image(path, split=split)
        if res is not None:
            r, h, b = res
            results_rlc.append(r)
            results_huff.append(h)
            results_byte.append(b)
            ok_count += 1
        else:
            err_count += 1
    return results_rlc, results_huff, results_byte, ok_count, err_count


def summarize_modality(all_lengths_list, prefix, cutoffs=[64, 96, 128, 160, 192, 196, 224, 256, 288, 320, 384, 512]):
    all_lengths_mcu = np.concatenate(all_lengths_list, axis=0) # Total MCUs
    total_mcus = all_lengths_mcu.size
    summary = percentile_dict(all_lengths_mcu, prefix)
    
    # MCU-level truncation
    trunc_dict = {}
    for L in cutoffs:
        gt_cnt = int((all_lengths_mcu > L).sum())
        ratio = float(gt_cnt / max(total_mcus, 1))
        trunc_dict[f"gt_{L}"] = gt_cnt
        trunc_dict[f"gt_{L}_ratio"] = ratio
        trunc_dict[f"gt_{L}_pct"] = ratio * 100.0
    summary[f"{prefix}_truncation"] = trunc_dict

    # Image-level truncation (at least 1 MCU truncated in the image)
    img_trunc_dict = {}
    for L in cutoffs:
        img_has_trunc = [int((img_mcus > L).any()) for img_mcus in all_lengths_list]
        img_gt_cnt = sum(img_has_trunc)
        img_ratio = float(img_gt_cnt / max(len(all_lengths_list), 1))
        img_trunc_dict[f"img_gt_{L}"] = img_gt_cnt
        img_trunc_dict[f"img_gt_{L}_ratio"] = img_ratio
        img_trunc_dict[f"img_gt_{L}_pct"] = img_ratio * 100.0
    summary[f"{prefix}_image_truncation"] = img_trunc_dict

    return summary, all_lengths_mcu


def main():
    parser = argparse.ArgumentParser(description="Stat full ImageNet validation set per-MCU lengths and truncation rates.")
    parser.add_argument("--index-path", default=str(REPO_ROOT / "assets" / "indexbase_val.csv"))
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (None for all 50,000)")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "val_full_truncation_stats.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    index_df = pd.read_csv(args.index_path)
    all_relpaths = index_df.iloc[:, 0].tolist()
    if args.limit is not None and args.limit > 0:
        all_relpaths = all_relpaths[:args.limit]

    all_imgpaths = [os.path.join(args.basepath, p) if args.basepath else p for p in all_relpaths]
    total_imgs = len(all_imgpaths)
    print(f"[{time.strftime('%X')}] Loaded {total_imgs} image paths from {args.index_path}")
    print(f"[{time.strftime('%X')}] Starting multiprocessing with {args.workers} workers, chunk_size={args.chunk_size}...")

    chunks = [all_imgpaths[i:i + args.chunk_size] for i in range(0, total_imgs, args.chunk_size)]
    
    all_rlc = []
    all_huff = []
    all_byte = []
    total_ok = 0
    total_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, chunk, args.split) for chunk in chunks]
        for idx, fut in enumerate(futures, start=1):
            r_list, h_list, b_list, ok, err = fut.result()
            all_rlc.extend(r_list)
            all_huff.extend(h_list)
            all_byte.extend(b_list)
            total_ok += ok
            total_err += err
            if idx % 20 == 0 or idx == len(chunks):
                elapsed = time.perf_counter() - t0
                speed = total_ok / max(elapsed, 1e-6)
                print(f"[{time.strftime('%X')}] Progress: {total_ok}/{total_imgs} images ({total_ok/total_imgs*100:.1f}%), {speed:.1f} img/s, errors={total_err}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[{time.strftime('%X')}] Processing complete in {elapsed:.1f}s. OK={total_ok}, Errors={total_err}")

    cutoffs = [64, 96, 128, 160, 192, 196, 224, 256, 288, 320, 384]
    
    rlc_summary, rlc_mcu = summarize_modality(all_rlc, "rlc", cutoffs)
    huff_summary, huff_mcu = summarize_modality(all_huff, "huffman", cutoffs)
    byte_summary, byte_mcu = summarize_modality(all_byte, "raw_byte", cutoffs)

    final_output = {
        "dataset": "ImageNet-1K Validation Full",
        "total_images": total_ok,
        "total_mcus": total_ok * 196,
        "elapsed_seconds": elapsed,
        "cutoffs": cutoffs,
        "raw_byte": byte_summary,
        "rlc": rlc_summary,
        "huffman": huff_summary,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"[{time.strftime('%X')}] Saved full stats to {args.out}")

    # Print markdown table
    print("\n" + "=" * 80)
    print("### ImageNet-1K Full Validation Set (50,000 Images = 9,800,000 MCUs) Truncation Summary")
    print("=" * 80)
    print(f"| Modality | Mean | Median(P50) | P90 | P95 | P99 | Max | Trunc@128 | Trunc@196 | Trunc@256 | Trunc@320 |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, s, prefix in [("Raw Bytes", byte_summary, "raw_byte"), ("RLC Events", rlc_summary, "rlc"), ("Parsed Huffman", huff_summary, "huffman")]:
        mean_v = s[f"{prefix}_mean"]
        p50_v = s[f"{prefix}_p50"]
        p90_v = s[f"{prefix}_p90"]
        p95_v = s[f"{prefix}_p95"]
        p99_v = s[f"{prefix}_p99"]
        max_v = s[f"{prefix}_p100"]
        t128 = s[f"{prefix}_truncation"]["gt_128_pct"]
        t196 = s[f"{prefix}_truncation"]["gt_196_pct"]
        t256 = s[f"{prefix}_truncation"]["gt_256_pct"]
        t320 = s[f"{prefix}_truncation"]["gt_320_pct"]
        unit = " B" if "Byte" in name else ""
        print(f"| {name} | {mean_v:.1f}{unit} | {p50_v:.0f}{unit} | {p90_v:.0f}{unit} | {p95_v:.0f}{unit} | {p99_v:.0f}{unit} | {max_v:.0f}{unit} | {t128:.2f}% | {t196:.2f}% | {t256:.2f}% | {t320:.4f}% |")
    print("=" * 80)


if __name__ == "__main__":
    main()
