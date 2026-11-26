"""
One-step memory smoke for RLC stages.
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import models.plainvit as plainvit
import datasets as ds
import utils.configs as configs
import utils.pipeline_utils as utils


def cuda_mem(prefix):
    if not torch.cuda.is_available():
        return f"{prefix}: cuda unavailable"
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"{prefix}: alloc={allocated:.2f}GiB peak={peak:.2f}GiB"


def synthetic_rlc_fields(batch_size, lmax, device):
    return {
        "kind": torch.randint(1, 5, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "component": torch.randint(1, 4, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "block": torch.randint(1, 7, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "zigzag": torch.randint(1, 65, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "run": torch.randint(1, 17, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "size": torch.randint(1, 17, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "amplitude": torch.randn(batch_size, 196, lmax, device=device) * 50,
        "mask": torch.ones(batch_size, 196, lmax, dtype=torch.bool, device=device),
    }


def synthetic_rlc_2d_fields(batch_size, lmax, device):
    run = torch.rand(batch_size, 196, lmax, device=device)
    value = torch.randn(batch_size, 196, lmax, device=device) * 0.5
    mask = torch.ones(batch_size, 196, lmax, dtype=torch.bool, device=device)
    return run, value, mask


def synthetic_rlc_cp_fields(batch_size, lmax, device):
    run = torch.randint(0, 16, (batch_size, 196, lmax), device=device, dtype=torch.long)
    value = torch.randn(batch_size, 196, lmax, device=device) * 0.5
    mask = torch.ones(batch_size, 196, lmax, dtype=torch.bool, device=device)
    return run, value, mask


def expected_patch_embed_type(stage):
    stage = ds.normalize_stage_name(stage)
    if stage in plainvit.RLC_CP_DIRECT_VARIANT_STAGES or stage in plainvit.HUFFMAN_CP_DIRECT_VARIANT_STAGES:
        return plainvit.PatchEmbedding_EventCPFusion
    if stage == "huff_dcac_2d_cp_direct":
        return plainvit.PatchEmbedding_HuffmanEvent_CP
    if stage in ("huff_dcac_2d_direct",):
        return plainvit.PatchEmbedding_RLCEvent_2D
    if stage in ("huff_dcac_flat_direct",):
        return plainvit.PatchEmbedding_RLCEvent_Flat
    if stage in ("rlc_dc_2d_direct", "rlc_dcac_2d_direct", "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct"):
        return plainvit.PatchEmbedding_RLCEvent_2D
    if stage in ("rlc_dc_flat_direct", "rlc_dcac_flat_direct"):
        return plainvit.PatchEmbedding_RLCEvent_Flat
    if stage in ("rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct"):
        return plainvit.PatchEmbedding_RLCEvent_CP
    if stage in ("rlc_dc_attnpool", "rlc_dcac_attnpool"):
        return plainvit.PatchEmbedding_RLCEvent_AttnPool
    if stage in ("rlc_dc_simple", "rlc_dcac_simple"):
        return plainvit.PatchEmbedding_RLCEvent_Simple
    if stage in ("rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max"):
        return plainvit.PatchEmbedding_RLCEvent_2D
    if stage in ("rlc_dc_flat", "rlc_dcac_flat"):
        return plainvit.PatchEmbedding_RLCEvent_Flat
    if stage in ("rlc_dc_2d_cp", "rlc_dcac_2d_cp"):
        return plainvit.PatchEmbedding_RLCEvent_CP
    if stage in ("rlc_dc", "rlc_dcac"):
        return plainvit.PatchEmbedding_RLCEvent
    raise ValueError(f"Unsupported RLC stage: {stage}")


def run_one_stage(stage, args, device):
    stage = ds.normalize_stage_name(stage)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg = configs.generate_config(
        modelarch="vits",
        stage=stage,
        classes=args.classes,
        batchsize=args.batch,
        epochs=1,
        scheduler_epochs=1,
        lr=args.lr,
        wd=args.wd,
        warmup_steps=args.warmup_steps,
        num_ops=args.num_ops,
        ops_magnitude=args.ops_magnitude,
        amp=args.amp,
        ampdtype=args.ampdtype,
        rlc_lmax=args.rlc_lmax,
    )
    cfg = utils.update_config(
        cfg, rank=0, world_size=1, threads=args.num_workers,
        run_train=True, run_eval=False, deterministic=False,
    )
    model = utils.get_model(cfg, report=False).to(device)
    model.train()
    expected_type = expected_patch_embed_type(stage)
    assert isinstance(model.patchembed, expected_type), (
        f"{stage}: expected {expected_type.__name__}, got {type(model.patchembed).__name__}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lmax = cfg.MODEL.INPUTSIZE[-1] if isinstance(cfg.MODEL.INPUTSIZE, tuple) else cfg.MODEL.RLC_LMAX
    fields = utils.make_synthetic_stage_input(cfg, batch_size=args.batch, device=device)
    labels = torch.randint(0, args.classes, (args.batch,), device=device)

    print(f"=== {stage} batch={args.batch} lmax={lmax} amp={args.amp} ===")
    print(f"  {cuda_mem('before forward')}")
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    if args.amp and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if args.ampdtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(fields)
            loss = criterion(outputs, labels)
    else:
        outputs = model(fields)
        loss = criterion(outputs, labels)
    print(f"  forward={time.time() - t0:.2f}s out={tuple(outputs.shape)} loss={loss.item():.4f}")
    print(f"  {cuda_mem('after forward')}")

    t1 = time.time()
    loss.backward()
    optimizer.step()
    print(f"  backward+step={time.time() - t1:.2f}s")
    print(f"  {cuda_mem('after step')}")

    del model, optimizer, fields, outputs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="rlc_dc_attnpool,rlc_dcac_attnpool")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--rlc_lmax", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_ops", type=int, default=2)
    parser.add_argument("--ops_magnitude", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ampdtype", default="bf16", choices=("bf16", "fp16"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"device={device}, gpu={gpu}")
    for stage in [item.strip() for item in args.stages.split(",") if item.strip()]:
        run_one_stage(stage, args, device)
    print("\nALL SMOKES PASSED")


if __name__ == "__main__":
    main()
