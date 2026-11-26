#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import torch

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise SystemExit("tensorboard is required to extract scalar metrics") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import datasets as ds
import models.plainvit as pvit
from utils.configs import generate_config


METRIC_TAGS = (
    "Acc/Test",
    "Acc/TestTop5",
    "Acc/Val",
    "Acc/ValTop5",
    "Acc/Train_val",
    "Acc/Train_valTop5",
    "Loss/Test",
    "Loss/Val",
    "Loss/Train_val",
    "Loss/Train",
    "Learning Rate",
)


def scalar_series(accumulator, tag):
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [{"step": int(event.step), "value": float(event.value)} for event in accumulator.Scalars(tag)]


def summarize_series(series, prefer="max"):
    if not series:
        return None
    key = (lambda item: item["value"])
    best = max(series, key=key) if prefer == "max" else min(series, key=key)
    return {
        "count": len(series),
        "first_step": series[0]["step"],
        "first": series[0]["value"],
        "last_step": series[-1]["step"],
        "last": series[-1]["value"],
        "best_step": best["step"],
        "best": best["value"],
    }


def run_name_from_tb_dir(path):
    name = Path(path).name
    return re.sub(r"_lr[0-9E+\-.]+_wd[0-9E+\-.]+_drop.*$", "", name)


def infer_stage(run_name):
    lowered = run_name.lower()
    aliases = sorted(set(ds.STAGE_ALIASES.values()), key=len, reverse=True)
    for stage in aliases:
        if stage in lowered:
            return stage
    if "rgb" in lowered:
        return "rgb"
    if "dct" in lowered:
        return "dct_float"
    return ""


def parse_lmax(run_name, default=320):
    match = re.search(r"lmax(\d+)", run_name)
    if match:
        return int(match.group(1))
    return default


def parse_arch(run_name):
    lowered = run_name.lower()
    if "vitb" in lowered:
        return "vitb"
    if "vitl" in lowered:
        return "vitl"
    if "vitti" in lowered:
        return "vitti"
    return "vits"


def model_param_summary(stage, run_name, lmax):
    if not stage:
        return {}
    try:
        cfg = generate_config(
            modelarch=parse_arch(run_name),
            stage=stage,
            rlc_lmax=lmax,
            batchsize=1024,
            amp=True,
            ampdtype="bf16",
        )
        model = pvit.ViT(
            in_channels=3,
            patch_size=cfg.MODEL.PATCHSIZE,
            emb_size=cfg.MODEL.EMBEDSIZE,
            depth=cfg.MODEL.DEPTH,
            n_classes=cfg.MODEL.CLASSES,
            drop_p=cfg.TRAIN.DROP,
            device="cpu",
            dtype=torch.float32,
            num_heads=cfg.MODEL.HEADS,
            head_size=cfg.MODEL.HEADSIZE,
            pixel_space=cfg.MODEL.DOMAIN,
            ver=cfg.MODEL.VERSION,
            use_subblock=cfg.MODEL.SUBBLOCK,
            stage=cfg.MODEL.STAGE,
            numeric_feature_dim=cfg.MODEL.NUMERIC_FEATURE_DIM,
            rlc_lmax=cfg.MODEL.RLC_LMAX,
        )
    except Exception as exc:
        return {"param_error": f"{type(exc).__name__}: {exc}"}
    total = sum(param.numel() for param in model.parameters())
    patch = sum(param.numel() for param in model.patchembed.parameters())
    return {
        "stage": stage,
        "arch": cfg.MODEL.ARCH,
        "numeric_feature_dim": int(cfg.MODEL.NUMERIC_FEATURE_DIM),
        "rlc_lmax": int(cfg.MODEL.RLC_LMAX),
        "total_params": int(total),
        "patchembed_params": int(patch),
        "non_patchembed_params": int(total - patch),
    }


def summarize_tb_dir(path):
    events = sorted(glob(os.path.join(path, "events.out.tfevents.*")))
    if not events:
        return None
    accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
    accumulator.Reload()
    run_name = run_name_from_tb_dir(path)
    stage = infer_stage(run_name)
    lmax = parse_lmax(run_name)
    summary = {
        "run": run_name,
        "tensorboard_dir": path,
        "event_files": len(events),
        "stage": stage,
        "rlc_lmax_in_name": lmax,
    }
    for tag in METRIC_TAGS:
        series = scalar_series(accumulator, tag)
        prefer = "min" if tag.startswith("Loss/") else "max"
        tag_summary = summarize_series(series, prefer=prefer)
        if tag_summary is None:
            continue
        safe_tag = tag.replace("/", "_")
        for key, value in tag_summary.items():
            summary[f"{safe_tag}_{key}"] = value
    summary.update(model_param_summary(stage, run_name, lmax))
    if "Acc_Val_last" in summary and "Acc_Train_val_last" in summary:
        summary["train_val_gap_last"] = summary["Acc_Train_val_last"] - summary["Acc_Val_last"]
    return summary


def discover_tb_dirs(roots):
    dirs = []
    for root in roots:
        if os.path.isfile(root):
            continue
        for path in sorted(glob(os.path.join(root, "*"))):
            if os.path.isdir(path) and glob(os.path.join(path, "events.out.tfevents.*")):
                dirs.append(path)
    return dirs


def write_csv(rows, path):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract Phase A scalar metrics and model capacity from TensorBoard runs.")
    parser.add_argument("--tb-root", action="append", default=[str(REPO_ROOT / "models" / "tensorboard")])
    parser.add_argument("--out-json", default=str(REPO_ROOT / "outputs" / "phase_a" / "metrics_summary.json"))
    parser.add_argument("--out-csv", default=str(REPO_ROOT / "outputs" / "phase_a" / "metrics_summary.csv"))
    args = parser.parse_args()

    rows = []
    for tb_dir in discover_tb_dirs(args.tb_root):
        row = summarize_tb_dir(tb_dir)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: item["run"])

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_csv(rows, out_csv)
    print(f"wrote {len(rows)} runs")
    print(f"json: {out_json}")
    print(f"csv: {out_csv}")


if __name__ == "__main__":
    main()
