"""
Dump TensorBoard scalars from stagewise + main repo baselines.

Reads event files from:
  - DDViT_stagewise_jpeg: quant_dct, dc_diff_ac (both current and _numericstage archived)
  - DDViT: teacher_vits_rgb_ep90_corrected, probe_vits_dct_lr1e3_wd1e4_ep90_2h, probe_vits_dct_lr1e3_wd1e4_ep90_2h_bestval_ep87

For each run, prints:
  - total epochs observed
  - first/latest/best Val Acc
  - first/latest Val Loss
  - train loss at latest point
"""
import os
import sys
from glob import glob

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("need: pip install tensorboard")
    sys.exit(1)


RUNS = {
    # stagewise active
    "stagewise quant_dct (dctgroup, live)": "/path/to/DDViT_stagewise_jpeg/models/tensorboard/stagewise_vits_quant_dct_lr1e3_wd1e4_ep90_lr1E-03_wd1E-04_drop0.0",
    "stagewise dc_diff_ac (dctgroup, live)": "/path/to/DDViT_stagewise_jpeg/models/tensorboard/stagewise_vits_dc_diff_ac_lr1e3_wd1e4_ep90_lr1E-03_wd1E-04_drop0.0",
    # stagewise archived (NumericStage)
    "stagewise quant_dct (numericstage, archived)": "/path/to/DDViT_stagewise_jpeg/models/tensorboard/stagewise_vits_quant_dct_lr1e3_wd1e4_ep90_lr1E-03_wd1E-04_drop0.0_numericstage",
    "stagewise dc_diff_ac (numericstage, archived)": "/path/to/DDViT_stagewise_jpeg/models/tensorboard/stagewise_vits_dc_diff_ac_lr1e3_wd1e4_ep90_lr1E-03_wd1E-04_drop0.0_numericstage",
    # main baselines
    "main DCT baseline (probe ep90)":    "/path/to/DDViT/models/tensorboard/probe_vits_dct_lr1e3_wd1e4_ep90_2_lr1E-03_wd1E-04_drop0.0",
    "main DCT baseline (bestval ep87)":  "/path/to/DDViT/models/tensorboard/probe_vits_dct_lr1e3_wd1e4_ep90_2h_bestval_ep87_lr1E-03_wd1E-04_drop0.0",
    "main RGB teacher":                  "/path/to/DDViT/models/tensorboard/teacher_vits_rgb_ep90_corrected_lr1E-03_wd1E-04_drop0.0",
}


def scalars(acc, tag):
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in acc.Scalars(tag)]


def summarize(name, path):
    if not os.path.isdir(path):
        print(f"{name}\n  path missing: {path}\n")
        return
    events = sorted(glob(os.path.join(path, "events.out.tfevents.*")))
    if not events:
        print(f"{name}\n  no event files: {path}\n")
        return
    print(f"=== {name} ===")
    print(f"  path: {path}")
    print(f"  event files: {len(events)}")
    acc = EventAccumulator(path, size_guidance={"scalars": 0})  # 0 = no truncation
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if not tags:
        print("  no scalar tags found")
        print()
        return
    print(f"  tags: {sorted(tags)[:10]}{'...' if len(tags) > 10 else ''}")

    # Try common tag names
    for tag_candidates in [
        ("Val Acc", "val_acc", "Acc/Val", "Accuracy/Val", "Val/Acc"),
        ("Val Loss", "val_loss", "Loss/Val", "Val/Loss"),
        ("Train Loss", "train_loss", "Loss/Train", "Loss/Peritr_Train", "Loss/EpochEnd_Train"),
    ]:
        series = None
        tag_used = None
        for t in tag_candidates:
            if t in tags:
                series = scalars(acc, t)
                tag_used = t
                break
        if series:
            best = max(series, key=lambda xy: xy[1]) if "Acc" in tag_used else min(series, key=lambda xy: xy[1])
            print(f"  [{tag_used}]  n={len(series)}  first=(step={series[0][0]}, v={series[0][1]:.4f})  "
                  f"latest=(step={series[-1][0]}, v={series[-1][1]:.4f})  "
                  f"best=(step={best[0]}, v={best[1]:.4f})")
        else:
            print(f"  [{'/'.join(tag_candidates)}]  not found")
    print()


def main():
    for name, path in RUNS.items():
        summarize(name, path)


if __name__ == "__main__":
    main()
