"""
One-step memory smoke for the current stage-wise quant_dct / dc_diff_ac path.

Default mode uses synthetic tensors at the formal per-GPU shape:
    batch=1024, ViT-S, PatchEmbedding_DCT_Group, fp32, no AMP.

Optional real-data mode runs one DataLoader batch, useful for catching collate /
CPU transform issues before submitting long PBS jobs.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import datasets as ds
import models.plainvit as plainvit
import utils.configs as configs
import utils.pipeline_utils as utils


def cuda_mem(prefix):
    if not torch.cuda.is_available():
        return f"{prefix}: cuda unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"{prefix}: alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"


def build_model(stage, args, device):
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
        amp=1 if args.amp else 0,
        ampdtype=args.ampdtype,
    )
    cfg = utils.update_config(
        cfg,
        rank=0,
        world_size=1,
        threads=args.num_workers,
        run_train=True,
        run_eval=False,
        deterministic=False,
    )
    model = utils.get_model(cfg, report=False).to(device)
    assert isinstance(model.patchembed, plainvit.PatchEmbedding_DCT_Group), (
        stage,
        type(model.patchembed).__name__,
    )
    return cfg, model


def synthetic_batch(batch_size, classes, device):
    data = {
        "y": torch.randn(batch_size, 1, 28, 28, 8, 8, device=device) / 189.0,
        "cbcr": torch.randn(batch_size, 2, 14, 14, 8, 8, device=device) / 189.0,
    }
    labels = torch.randint(0, classes, (batch_size,), device=device)
    return data, labels


def real_batch(stage, args, device):
    dataset = ds.imagenet_dataset_indexing(
        indexfile=args.index,
        basepath=args.data,
        load_mode=stage,
        stage=stage,
        type="train",
        ops_list=None,
        num_ops=args.num_ops,
        ops_magnitude=args.ops_magnitude,
        dtype=torch.float32,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=None if args.num_workers == 0 else 2,
        collate_fn=ds.optimized_stage_collate_fn,
    )
    data, labels = next(iter(loader))
    return utils.move_to_device(data, device), labels.to(device, non_blocking=True)


def run_one_stage(stage, args, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg, model = build_model(stage, args, device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, eps=1e-8)
    weight_decayer = utils.coptim.WeightDecay(
        [p for name, p in model.named_parameters() if (".weight" in name) and ("lrnorm" not in name)],
        lr=args.lr,
        weight_decay=args.wd,
    )

    t0 = time.time()
    if args.real_data:
        data, labels = real_batch(stage, args, device)
        source = "real"
    else:
        data, labels = synthetic_batch(args.batch, args.classes, device)
        source = "synthetic"
    t_data = time.time() - t0

    print(f"=== {stage} ===")
    print(f"  source={source} batch={args.batch} amp={args.amp} ampdtype={args.ampdtype}")
    print(f"  y={tuple(data['y'].shape)} cbcr={tuple(data['cbcr'].shape)} labels={tuple(labels.shape)}")
    print(f"  patchembed={type(model.patchembed).__name__} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"  data_time={t_data:.2f}s")
    print("  " + cuda_mem("before forward"))

    optimizer.zero_grad(set_to_none=True)
    weight_decayer.zero_grad(set_to_none=True)
    t1 = time.time()
    with torch.amp.autocast(device_type=device, enabled=args.amp, dtype=utils.get_dtype(args.ampdtype)):
        outputs = model(data)
        loss = criterion(outputs, labels)
    t_forward = time.time() - t1
    print(f"  out={tuple(outputs.shape)} loss={loss.item():.4f} forward={t_forward:.2f}s")
    print("  " + cuda_mem("after forward"))

    t2 = time.time()
    if args.amp:
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.step(weight_decayer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        weight_decayer.step()
    t_backward = time.time() - t2
    grad_params = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in model.parameters())
    print(f"  backward+step={t_backward:.2f}s grad_params={grad_params}/{total_params}")
    print("  " + cuda_mem("after step"))
    print("  OK")

    del model, optimizer, weight_decayer, data, labels, outputs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="quant_dct,dc_diff_ac")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_ops", type=int, default=2)
    parser.add_argument("--ops_magnitude", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ampdtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument("--data", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--index", default=os.path.join(REPO, "assets", "indexbase_train.csv"))
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}")
    print(f"cwd={os.getcwd()}")
    print(f"real_data={args.real_data}")
    if args.real_data:
        assert os.path.isdir(args.data), f"missing data dir: {args.data}"
        assert os.path.isfile(args.index), f"missing index: {args.index}"

    for stage in [item.strip() for item in args.stages.split(",") if item.strip()]:
        run_one_stage(stage, args, device)
    print("\nONE-STEP MEMORY SMOKE PASSED")


if __name__ == "__main__":
    main()
