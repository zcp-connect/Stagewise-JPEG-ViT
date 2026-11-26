#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torchmetrics

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets as ds
from utils import configs
import utils.pipeline_utils as utils


STAGE_SPECS = {
    "rlc_dcac_2d_direct": {
        "savepath": "models/stagewise_vits_rlc_dcac_2d_direct_lmax320_ep90.pth",
        "numeric_feature_dim": 64,
        "amp": True,
        "ampdtype": "bf16",
    },
    "huff_dcac_2d_direct": {
        "savepath": "models/stagewise_vits_huff_dcac_2d_direct_lmax320_ep90.pth",
        "numeric_feature_dim": 64,
        "amp": True,
        "ampdtype": "bf16",
    },
    "rlc_dcac_2d_cp_concat_conv_direct_d128": {
        "stage": "rlc_dcac_2d_cp_concat_conv_direct",
        "savepath": "models/stagewise_vits_rlc_dcac_2d_cp_concat_conv_direct_d128_lmax320_ep90.pth",
        "numeric_feature_dim": 128,
        "amp": True,
        "ampdtype": "bf16",
    },
    "huff_dcac_2d_cp_concat_conv_direct_d128": {
        "stage": "huff_dcac_2d_cp_concat_conv_direct",
        "savepath": "models/stagewise_vits_huff_dcac_2d_cp_concat_conv_direct_d128_lmax320_ep90.pth",
        "numeric_feature_dim": 128,
        "amp": True,
        "ampdtype": "bf16",
    },
    "raw_byte_discrete_conv": {
        "stage": "raw_byte",
        "savepath": "models/stagewise_vits_raw_byte_discrete_conv_lmax320_ep90.pth",
        "numeric_feature_dim": 64,
        "amp": False,
        "ampdtype": "bf16",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lmax truncation at inference by zeroing/masking event positions > K."
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGE_SPECS),
        help="Stage alias to evaluate. Repeatable. Defaults to all four requested stages.",
    )
    parser.add_argument("--lmax", type=int, nargs="+", default=[128, 196, 256, 320])
    parser.add_argument("--datapath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--indexpath", default="assets/indexbase_val.csv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--in_order", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", default="outputs/lmax_inference_truncation/results.csv")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, tuple):
        return tuple(_move_to_device(item, device) for item in data)
    if isinstance(data, list):
        return [_move_to_device(item, device) for item in data]
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    return data


def truncate_event_batch(inputs, keep_lmax):
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        payload, mask = inputs
        if payload.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                "Expected 3D raw-byte tensors [B, MCU, L], got "
                f"{tuple(payload.shape)}, {tuple(mask.shape)}"
            )
        full_lmax = payload.shape[-1]
        if keep_lmax > full_lmax:
            raise ValueError(f"Requested keep_lmax={keep_lmax} exceeds input length {full_lmax}")
        if keep_lmax == full_lmax:
            return inputs
        payload = payload.clone()
        mask = mask.clone()
        payload[..., keep_lmax:] = 0
        mask[..., keep_lmax:] = False
        return payload, mask
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
        raise TypeError(f"Expected event tuple (symbol/run, amplitude, mask) or (payload, mask), got {type(inputs)}")
    symbol, amplitude, mask = inputs
    if symbol.ndim != 3 or amplitude.ndim != 3 or mask.ndim != 3:
        raise ValueError(
            "Expected 3D event tensors [B, MCU, L], got "
            f"{tuple(symbol.shape)}, {tuple(amplitude.shape)}, {tuple(mask.shape)}"
        )
    full_lmax = symbol.shape[-1]
    if keep_lmax > full_lmax:
        raise ValueError(f"Requested keep_lmax={keep_lmax} exceeds input length {full_lmax}")
    if keep_lmax == full_lmax:
        return inputs

    symbol = symbol.clone()
    amplitude = amplitude.clone()
    mask = mask.clone()
    symbol[..., keep_lmax:] = 0
    amplitude[..., keep_lmax:] = 0
    mask[..., keep_lmax:] = False
    return symbol, amplitude, mask


def make_cfg(stage, spec, args):
    cfg = configs.generate_config(
        modelarch="vits",
        domain="mcu_bytes",
        stage=spec.get("stage", stage),
        modelver=3,
        subblock=False,
        numeric_feature_dim=spec["numeric_feature_dim"],
        rlc_lmax=320,
        batchsize=args.batch_size,
        amp=1 if spec.get("amp", False) else 0,
        ampdtype=spec.get("ampdtype", "bf16"),
        prefetch_factor=args.prefetch_factor,
        in_order=bool(args.in_order),
    )
    cfg.RANK = 0
    cfg.WORLDSIZE = 1
    cfg.THREADS = args.num_workers
    return cfg


def load_model(cfg, savepath, device):
    model = utils.get_model(cfg, report=False).to(device)
    state = torch.load(savepath, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_loader(cfg, args):
    return ds.dataset_selector(
        dataset=cfg.TRAIN.DATASET,
        type="test",
        indexpath=args.indexpath,
        basepath=args.datapath,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        distributed=False,
        rank=0,
        world_size=1,
        seed=cfg.SEED,
        freqnorm=cfg.TRAIN.FREQNORM,
        freqnorm_stats=cfg.TRAIN.FREQNORM_STATS,
        rlc_lmax=320,
        prefetch_factor=args.prefetch_factor,
        in_order=bool(args.in_order),
    )


def evaluate_stage(stage_name, spec, keep_lmax_values, args, device):
    cfg = make_cfg(stage_name, spec, args)
    savepath = str((REPO_ROOT / spec["savepath"]).resolve())
    if not os.path.exists(savepath):
        raise FileNotFoundError(f"Missing checkpoint for {stage_name}: {savepath}")

    loader = make_loader(cfg, args)
    model = load_model(cfg, savepath, device)
    criterion = torch.nn.CrossEntropyLoss()
    meters = {}
    for keep_lmax in keep_lmax_values:
        meters[keep_lmax] = {
            "loss": 0.0,
            "batches": 0,
            "samples": 0,
            "top1": torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.MODEL.CLASSES, top_k=1
            ).to(device),
            "topk": (
                torchmetrics.Accuracy(
                    task="multiclass", num_classes=cfg.MODEL.CLASSES, top_k=args.topk
                ).to(device)
                if args.topk > 1 else None
            ),
        }
    amp_dtype = utils.get_dtype(cfg.MODEL.AMPDTYPE)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = _move_to_device(inputs, device)
            labels = labels.to(device, non_blocking=True)
            batch_size = int(labels.shape[0])
            for keep_lmax in keep_lmax_values:
                truncated_inputs = truncate_event_batch(inputs, keep_lmax)
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP, dtype=amp_dtype):
                    outputs = model(truncated_inputs)
                    loss = criterion(outputs, labels)
                meter = meters[keep_lmax]
                meter["loss"] += float(loss.item())
                meter["batches"] += 1
                meter["samples"] += batch_size
                meter["top1"].update(outputs, labels)
                if meter["topk"] is not None:
                    meter["topk"].update(outputs, labels)
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[{stage_name} keep_lmax={','.join(map(str, keep_lmax_values))}] "
                    f"{batch_idx + 1}/{len(loader)} batches",
                    flush=True,
                )

    results = []
    for keep_lmax in keep_lmax_values:
        meter = meters[keep_lmax]
        result = {
            "stage": stage_name,
            "model_stage": cfg.MODEL.STAGE,
            "trained_lmax": 320,
            "eval_keep_lmax": keep_lmax,
            "samples": meter["samples"],
            "loss": meter["loss"] / max(meter["batches"], 1),
            "top1": float(meter["top1"].compute().item()),
        }
        if meter["topk"] is not None:
            result[f"top{args.topk}"] = float(meter["topk"].compute().item())
        results.append(result)
    return results


def append_result(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "model_stage", "trained_lmax", "eval_keep_lmax", "samples", "loss", "top1"]
    topk_keys = sorted(key for key in result if key.startswith("top") and key != "top1")
    fieldnames.extend(topk_keys)
    exists = output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def main():
    args = parse_args()
    stages = args.stage or sorted(STAGE_SPECS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This full-val eval must run on a GPU node, not the login CPU node.")
    print(
        json.dumps(
            {
                "stages": stages,
                "lmax": args.lmax,
                "datapath": args.datapath,
                "indexpath": args.indexpath,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "output": args.output,
            },
            indent=2,
        ),
        flush=True,
    )

    keep_lmax_values = sorted(dict.fromkeys(int(value) for value in args.lmax))
    for stage_name in stages:
        spec = STAGE_SPECS[stage_name]
        for result in evaluate_stage(stage_name, spec, keep_lmax_values, args, device):
            append_result(args.output, result)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
