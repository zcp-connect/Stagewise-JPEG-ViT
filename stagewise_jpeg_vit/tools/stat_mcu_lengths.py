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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets as ds


def percentile_dict(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    out = {f"{prefix}_p{str(q).replace('.', '_')}": float(np.percentile(values, q)) for q in qs}
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


def collect_one(imgpath, split, dtype, lmax, use_randaugment, ops_list, num_ops, ops_magnitude, want_byte):
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
    _run, _value, _mask, event_len = ds.pack_rlc_event_tensors_direct(
        blocks_zz,
        lmax=lmax,
        dc_only=False,
        dtype=dtype,
        cp=False,
    )
    event_len_np = event_len.cpu().numpy().astype(np.int32)
    truncated = event_len_np >= lmax

    byte_len_np = None
    valid_bits_np = None
    if want_byte:
        new_dim = torch.tensor([[224, 224]], dtype=torch.int32)
        jpeg_buffer = ds.dm.write_coefficients_to_buffer(new_dim, _quant, y_out, cbcr_out)
        records = ds.bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)["mcus"]
        byte_len_np = np.asarray([int(m["byte_len"]) for m in records], dtype=np.int32)
        valid_bits_np = np.asarray([int(m.get("valid_bits", int(m["byte_len"]) * 8)) for m in records], dtype=np.int32)
    return event_len_np, truncated, byte_len_np, valid_bits_np


def main():
    parser = argparse.ArgumentParser(description="Stat per-MCU RLC event length and JPEG byte length.")
    parser.add_argument("--index-path", default="assets/indexbase_train.csv")
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=320)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--num-ops", type=int, default=2)
    parser.add_argument("--ops-magnitude", type=int, default=3)
    parser.add_argument("--byte", action="store_true", help="also compute actual entropy payload byte length")
    parser.add_argument("--out", default="outputs/mcu_length_stats.json")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    index = pd.read_csv(args.index_path).to_numpy()
    chosen = sample_indices(len(index), args.limit, args.seed)

    event_lengths = []
    byte_lengths = []
    valid_bits = []
    trunc_count = 0
    image_count = 0
    error_count = 0
    t0 = time.perf_counter()

    for j, idx in enumerate(chosen, start=1):
        relpath = index[idx, 0]
        imgpath = resolve_path(args.basepath, relpath)
        try:
            event_len, truncated, byte_len, valid_bit = collect_one(
                imgpath,
                args.split,
                dtype,
                args.lmax,
                args.randaugment,
                None,
                args.num_ops,
                args.ops_magnitude,
                args.byte,
            )
        except Exception as exc:
            error_count += 1
            if error_count <= 10:
                print(f"[error] {imgpath}: {type(exc).__name__}: {exc}", flush=True)
            continue
        image_count += 1
        event_lengths.append(event_len)
        trunc_count += int(truncated.sum())
        if byte_len is not None:
            byte_lengths.append(byte_len)
        if valid_bit is not None:
            valid_bits.append(valid_bit)
        if args.progress_every > 0 and j % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"processed={j}/{len(chosen)} ok_images={image_count} elapsed={elapsed:.1f}s imgs/s={image_count/max(elapsed,1e-9):.2f}", flush=True)

    if not event_lengths:
        raise RuntimeError("No samples processed successfully")

    event_lengths = np.concatenate(event_lengths)
    result = {
        "index_path": args.index_path,
        "basepath": args.basepath,
        "split": args.split,
        "limit_requested": args.limit,
        "images_seen": int(len(chosen)),
        "images_ok": int(image_count),
        "errors": int(error_count),
        "mcus": int(event_lengths.size),
        "lmax": int(args.lmax),
        "randaugment": bool(args.randaugment),
        "truncated_mcus_at_lmax": int(trunc_count),
        "truncated_ratio_at_lmax": float(trunc_count / max(event_lengths.size, 1)),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    result.update(percentile_dict(event_lengths, "event_len"))
    result["event_len_hist"] = length_hist(event_lengths, [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192), (192, 224), (224, 256), (256, 288), (288, 320), (320, None)])
    for cutoff in [64, 96, 128, 160, 192, 224, 256, 288, 320, 384]:
        result[f"event_len_gt_{cutoff}"] = int((event_lengths > cutoff).sum())
        result[f"event_len_gt_{cutoff}_ratio"] = float((event_lengths > cutoff).mean())

    if byte_lengths:
        byte_lengths = np.concatenate(byte_lengths)
        result.update(percentile_dict(byte_lengths, "byte_len"))
        result["byte_len_hist"] = length_hist(byte_lengths, [(0, 16), (16, 32), (32, 48), (48, 64), (64, 80), (80, 96), (96, 128), (128, 160), (160, 192), (192, 256), (256, None)])
        corr = np.corrcoef(event_lengths.astype(np.float64), byte_lengths.astype(np.float64))[0, 1]
        result["event_byte_corr"] = float(corr)
        for cutoff in [32, 48, 64, 80, 96, 128, 160, 192, 256]:
            result[f"byte_len_gt_{cutoff}"] = int((byte_lengths > cutoff).sum())
            result[f"byte_len_gt_{cutoff}_ratio"] = float((byte_lengths > cutoff).mean())
    if valid_bits:
        valid_bits = np.concatenate(valid_bits)
        result.update(percentile_dict(valid_bits, "valid_bits"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
