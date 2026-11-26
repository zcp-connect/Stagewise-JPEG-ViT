import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets as ds
from models.plainvit import ViT


def _batch_to_device(batch, device):
    data, target = batch
    if isinstance(data, (tuple, list)):
        data = tuple(t.to(device, non_blocking=True) for t in data)
    elif isinstance(data, dict):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        data = data.to(device, non_blocking=True)
    return data, target.to(device, non_blocking=True)


def test_dataset(args):
    train_index = args.index.split(",")[0]
    dataset = ds.imagenet_dataset_indexing(
        indexfile=train_index,
        basepath=args.data,
        load_mode="event_amp_only_flat_direct",
        stage="event_amp_only_flat_direct",
        type="train",
        rlc_lmax=args.rlc_lmax,
        ops_list=None,
        num_ops=args.num_ops,
        ops_magnitude=args.ops_magnitude,
        dtype=torch.float32,
    )
    data, label = dataset[0]
    assert isinstance(data, (tuple, list)), type(data)
    assert len(data) == 3, len(data)
    prefix, amplitude, mask = data
    assert prefix.shape == (196, args.rlc_lmax), prefix.shape
    assert amplitude.shape == (196, args.rlc_lmax), amplitude.shape
    assert mask.shape == (196, args.rlc_lmax), mask.shape
    assert amplitude.dtype == torch.float32, amplitude.dtype
    assert mask.dtype == torch.bool, mask.dtype
    assert int(mask.sum()) > 0
    print(
        "DATASET OK",
        f"label={label}",
        f"amp_shape={tuple(amplitude.shape)}",
        f"valid={int(mask.sum())}",
        f"amp_range=[{float(amplitude.min()):.4f},{float(amplitude.max()):.4f}]",
        flush=True,
    )


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU smoke must run on a CUDA node"
    model = ViT(
        emb_size=384,
        depth=12,
        n_classes=1000,
        drop_p=0.0,
        stage="event_amp_only_flat_direct",
        rlc_lmax=args.rlc_lmax,
        device=device,
        dtype=torch.float32,
    ).to(device)
    prefix = torch.zeros(args.batch, 196, args.rlc_lmax, device=device)
    amplitude = torch.randn(args.batch, 196, args.rlc_lmax, device=device) * 0.5
    mask = torch.ones(args.batch, 196, args.rlc_lmax, device=device, dtype=torch.bool)
    target = torch.randint(0, 1000, (args.batch,), device=device)
    with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
        output = model((prefix, amplitude, mask))
        loss = F.cross_entropy(output, target)
    loss.backward()
    grad_params = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert output.shape == (args.batch, 1000), output.shape
    assert grad_params > 0
    print(
        "MODEL OK",
        f"out={tuple(output.shape)}",
        f"loss={float(loss.detach()):.4f}",
        f"grad_params={grad_params}/{total_params}",
        f"patch_params={sum(p.numel() for p in model.patchembed.parameters())}",
        flush=True,
    )


def test_dataloader(args):
    loader = ds.dataset_selector(
        batch_size=args.loader_batch,
        dataset="imagenet_event_amp_only_flat_direct",
        basepath=args.data,
        indexpath=args.index.split(",")[0],
        type="train",
        num_workers=args.num_workers,
        shuffle=True,
        trainval_split=-1,
        dtype=torch.float32,
        rlc_lmax=args.rlc_lmax,
        ops_list=None,
        num_ops=args.num_ops,
        ops_magnitude=args.ops_magnitude,
        prefetch_factor=args.prefetch_factor,
        in_order=bool(args.in_order),
    )
    device = torch.device("cuda")
    times = []
    iterator = iter(loader)
    for i in range(args.num_batches):
        start = time.perf_counter()
        batch = next(iterator)
        _batch_to_device(batch, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        print(f"DATALOADER batch={i + 1}/{args.num_batches} time={times[-1]:.3f}s", flush=True)
    print(
        "DATALOADER OK",
        f"mean={sum(times)/len(times):.3f}s",
        f"max={max(times):.3f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--index", default="assets/indexbase_train.csv,assets/indexbase_val.csv")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--loader_batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--in_order", type=int, default=1)
    parser.add_argument("--rlc_lmax", type=int, default=320)
    parser.add_argument("--num_ops", type=int, default=2)
    parser.add_argument("--ops_magnitude", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    assert Path(args.data).exists(), args.data
    test_dataset(args)
    test_dataloader(args)
    test_model(args)
    print("ALL EVENT AMP ONLY SMOKES PASSED", flush=True)


if __name__ == "__main__":
    main()
