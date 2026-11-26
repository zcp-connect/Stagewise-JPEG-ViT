import argparse
import os
import sys

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from datasets import (
    load_augmented_rlc_event_tensors_2d,
    load_augmented_rlc_event_tensors_cp,
    load_augmented_rlc_event_tensors_direct,
)


def _first_image(index_path, basepath):
    rows = pd.read_csv(index_path).to_numpy()
    path = str(rows[0, 0])
    return path if os.path.isabs(path) else os.path.join(basepath, path)


def _compare_2d(img_path, lmax):
    old_run, old_value, old_mask = load_augmented_rlc_event_tensors_2d(
        img_path, "rlc_dcac_2d", type="val", lmax=lmax,
    )
    new_run, new_value, new_mask = load_augmented_rlc_event_tensors_direct(
        img_path, "rlc_dcac_2d_direct", type="val", lmax=lmax,
        use_randaugment=False,
    )
    mask_equal = torch.equal(old_mask, new_mask)
    run_diff = (old_run - new_run).abs().max().item()
    value_diff = (old_value - new_value).abs().max().item()
    print("=== 2D parity ===")
    print(f"mask_equal={mask_equal}")
    print(f"run_max_diff={run_diff:.8g}")
    print(f"value_max_diff={value_diff:.8g}")
    print(f"old_event_len_head={old_mask.sum(-1)[:5].tolist()}")
    print(f"new_event_len_head={new_mask.sum(-1)[:5].tolist()}")
    assert mask_equal
    assert run_diff < 1e-6
    assert value_diff < 1e-6


def _compare_cp(img_path, lmax):
    old_run, old_value, old_mask = load_augmented_rlc_event_tensors_cp(
        img_path, "rlc_dcac_2d_cp", type="val", lmax=lmax,
    )
    new_run, new_value, new_mask = load_augmented_rlc_event_tensors_direct(
        img_path, "rlc_dcac_2d_cp_direct", type="val", lmax=lmax,
        use_randaugment=False,
    )
    mask_equal = torch.equal(old_mask, new_mask)
    run_equal = torch.equal(old_run, new_run)
    value_diff = (old_value - new_value).abs().max().item()
    print("=== CP parity ===")
    print(f"mask_equal={mask_equal}")
    print(f"run_equal={run_equal}")
    print(f"value_max_diff={value_diff:.8g}")
    print(f"old_event_len_head={old_mask.sum(-1)[:5].tolist()}")
    print(f"new_event_len_head={new_mask.sum(-1)[:5].tolist()}")
    assert mask_equal
    assert run_equal
    assert value_diff < 1e-6


def _randaug_sanity(img_path, lmax):
    run, value, mask = load_augmented_rlc_event_tensors_direct(
        img_path, "rlc_dcac_2d_direct", type="train", lmax=lmax,
        use_randaugment=True,
    )
    print("=== train RandAugment sanity ===")
    print(f"run_shape={tuple(run.shape)} dtype={run.dtype}")
    print(f"value_shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"mask_shape={tuple(mask.shape)} dtype={mask.dtype}")
    print(
        "event_len "
        f"min={int(mask.sum(-1).min())} "
        f"mean={mask.sum(-1).float().mean().item():.2f} "
        f"max={int(mask.sum(-1).max())}"
    )
    assert run.shape == (196, lmax)
    assert value.shape == (196, lmax)
    assert mask.shape == (196, lmax)
    assert mask.dtype == torch.bool
    assert int(mask.sum(-1).min()) >= 6
    assert int(mask.sum(-1).max()) <= lmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default=os.path.join(REPO, "assets", "indexbase_val.csv"))
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--image", default=None)
    parser.add_argument("--lmax", type=int, default=320)
    args = parser.parse_args()

    img_path = args.image or _first_image(args.index_path, args.basepath)
    print(f"image={img_path}")
    _compare_2d(img_path, args.lmax)
    _compare_cp(img_path, args.lmax)
    _randaug_sanity(img_path, args.lmax)


if __name__ == "__main__":
    main()
