#!/usr/bin/env python3
"""Event-position importance for flat RLC/Huffman tokenizers (Method A: weight-norm).

Reads the trained flat projection Linear(lmax*2 -> emb) and computes, for each of
the lmax event positions, the Frobenius norm of the two input columns (run, value)
that feed that position. This is a zero-forward-pass, prior/capacity view of which
event-stream positions the model allocates projection capacity to.

x-axis = event-stream position (NOT zigzag frequency; RLE makes position->frequency
data-dependent). Position 0 is always the Y0 DC term; later positions interleave
block-boundary DCs and AC (run,value) events across the 6 blocks (Y0..3, Cb, Cr).

Usage:
  python tools/event_importance_weightnorm.py \
      --ckpt models/stagewise_vits_rlc_dcac_flat_direct_lmax320_ep90.pth \
      --lmax 320 --out outputs/event_importance/rlc_flat_weightnorm.csv
"""
import argparse, os, csv
import torch


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # strip DDP 'module.' prefix if present
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def find_flat_projection(sd):
    cands = [k for k in sd if "patchembed" in k and "projection" in k and k.endswith("weight")]
    # flat projection is the 2D Linear weight with in_features == lmax*2 (or lmax for amplitude-flat)
    twod = [k for k in cands if sd[k].dim() == 2]
    if not twod:
        raise SystemExit(f"No 2D projection weight under patchembed. candidates={cands}")
    # prefer the largest in_features (the flat one)
    twod.sort(key=lambda k: sd[k].shape[1], reverse=True)
    return twod[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmax", type=int, default=320)
    ap.add_argument("--per_event_dims", type=int, default=2,
                    help="2 for RLC (run,value); 1 for amplitude-only flat")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = load_state_dict(args.ckpt)
    key = find_flat_projection(sd)
    W = sd[key].float()  # (emb, lmax*per_event_dims)
    emb, in_feat = W.shape
    d = args.per_event_dims
    expected = args.lmax * d
    print(f"using weight: {key}  shape=({emb},{in_feat})  lmax={args.lmax} per_event_dims={d}")
    if in_feat != expected:
        # infer per_event_dims if mismatch
        if in_feat == args.lmax:
            d = 1
        elif in_feat == args.lmax * 2:
            d = 2
        else:
            raise SystemExit(f"in_features={in_feat} != lmax*{d}={expected}; check lmax/per_event_dims")
        print(f"  adjusted per_event_dims -> {d}")

    Wg = W.view(emb, args.lmax, d)              # (emb, lmax, d)
    pos_norm = Wg.norm(dim=(0, 2))              # (lmax,) Frobenius over emb & per-event dims
    # also split run vs value contribution if d==2
    run_norm = Wg[:, :, 0].norm(dim=0) if d == 2 else None
    val_norm = Wg[:, :, 1].norm(dim=0) if d == 2 else Wg[:, :, 0].norm(dim=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        if d == 2:
            w.writerow(["position", "importance", "run_norm", "value_norm"])
            for k in range(args.lmax):
                w.writerow([k, float(pos_norm[k]), float(run_norm[k]), float(val_norm[k])])
        else:
            w.writerow(["position", "importance", "value_norm"])
            for k in range(args.lmax):
                w.writerow([k, float(pos_norm[k]), float(val_norm[k])])

    # quick textual summary (no matplotlib dependency)
    tot = pos_norm.sum().item()
    cum = torch.cumsum(pos_norm, 0) / tot
    p50 = int((cum < 0.5).sum())
    p90 = int((cum < 0.9).sum())
    p99 = int((cum < 0.99).sum())
    print(f"wrote {args.out}")
    print(f"importance mass: 50% within first {p50} positions, 90% within {p90}, 99% within {p99} (of {args.lmax})")
    print(f"first-8 positions importance: {[round(float(x),3) for x in pos_norm[:8]]}")
    print(f"last-8  positions importance: {[round(float(x),3) for x in pos_norm[-8:]]}")


if __name__ == "__main__":
    main()
