#!/usr/bin/env python3
"""Feature caching and linear-probe training for selected JPEG/RGB/raw-byte stages.

Heavy modes must run on GPU compute/interactive nodes, not login nodes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm

# STAGEWISE_REPO defaults to this repository's `stagewise_jpeg_vit/` directory.
# DATA_ROOT (ImageNet-1k root with train/ and val/) is configured via IMAGENET_ROOT.
# OLD_REPO (checkpoint/code root of the earlier RGB / raw-MCU-byte baselines, only needed for the
# `rgb` and `raw_byte` stages) is configured via DDVIT_OLD_REPO.
STAGEWISE_REPO = Path(os.environ.get('STAGEWISE_REPO', Path(__file__).resolve().parents[1]))
OLD_REPO = Path(os.environ.get('DDVIT_OLD_REPO', '/path/to/DDViT'))
DATA_ROOT = Path(os.environ.get('IMAGENET_ROOT', '/path/to/imagenet1k/imagenet1k'))
PLAIN_FREQNORM_STATS = STAGEWISE_REPO / 'outputs/dct_stats/imagenet1k_train_dct_freq_stats.pt'
if str(STAGEWISE_REPO) not in sys.path:
    sys.path.insert(0, str(STAGEWISE_REPO))


@dataclass(frozen=True)
class StageSpec:
    name: str
    backend: str
    stage: str
    checkpoint: Path
    e2e_top1: float
    e2e_top5: float | None
    freqnorm: bool = False
    freqnorm_stats: Path | None = None
    modelver: int = 1
    numeric_feature_dim: int | None = None
    rlc_lmax: int = 320
    note: str = ''


STAGE_SPECS = {
    'rgb': StageSpec(
        name='rgb', backend='old', stage='rgb',
        checkpoint=OLD_REPO / 'models/teacher_vits_rgb_ep90_corrected.pth',
        e2e_top1=76.508, e2e_top5=None,
        note='Old DDViT RGB baseline.',
    ),
    'dct_float_freqnorm': StageSpec(
        name='dct_float_freqnorm', backend='stagewise', stage='dct_float',
        checkpoint=STAGEWISE_REPO / 'models/stagewise_vits_dct_float_freqnorm_lr1e3_wd1e4_ep90.pth',
        e2e_top1=74.482, e2e_top5=91.910,
        freqnorm=True, freqnorm_stats=PLAIN_FREQNORM_STATS,
    ),
    'quant_dct_freqnorm': StageSpec(
        name='quant_dct_freqnorm', backend='stagewise', stage='quant_dct',
        checkpoint=STAGEWISE_REPO / 'models/stagewise_vits_quant_dct_freqnorm_lr1e3_wd1e4_ep90.pth',
        e2e_top1=74.338, e2e_top5=91.650,
        freqnorm=True, freqnorm_stats=PLAIN_FREQNORM_STATS,
    ),
    'rlc_conv': StageSpec(
        name='rlc_conv', backend='stagewise', stage='rlc_dcac_2d_cp_concat_conv_direct',
        checkpoint=STAGEWISE_REPO / 'models/stagewise_vits_rlc_dcac_2d_cp_concat_conv_direct_lmax320_ep90.pth',
        e2e_top1=68.874, e2e_top5=None,
        numeric_feature_dim=64, rlc_lmax=320,
    ),
    'rlc_2d_cp': StageSpec(
        name='rlc_2d_cp', backend='stagewise', stage='rlc_dcac_2d_cp_direct',
        checkpoint=STAGEWISE_REPO / 'models/stagewise_vits_rlc_dcac_2d_cp_direct_lmax320_ep90.pth',
        e2e_top1=45.000, e2e_top5=None,
        numeric_feature_dim=64, rlc_lmax=320,
        note='Collapsed mean-pool CP direct RLC control; E2E Top-1 rounded pending exact registry extraction.',
    ),
    'huff_conv': StageSpec(
        name='huff_conv', backend='stagewise', stage='huff_dcac_2d_cp_concat_conv_direct',
        checkpoint=STAGEWISE_REPO / 'models/stagewise_vits_huff_dcac_2d_cp_concat_conv_direct_lmax320_ep90.pth',
        e2e_top1=70.746, e2e_top5=None,
        numeric_feature_dim=64, rlc_lmax=320,
    ),
    'raw_byte': StageSpec(
        name='raw_byte', backend='old', stage='mcu_bytes',
        checkpoint=OLD_REPO / 'models/vits_mcu_bytes_sincos_ep90.pth',
        e2e_top1=48.894, e2e_top5=72.752,
        note='Old DDViT raw MCU byte baseline.',
    ),
}


class RepoContext:
    def __init__(self, repo: Path):
        self.repo = repo
        self._old_sys_path = None
        self._old_modules = None

    def __enter__(self):
        prefixes = ('datasets', 'models', 'utils', 'eval')
        self._old_modules = {k: v for k, v in sys.modules.items() if k == 'datasets' or k == 'eval' or k.startswith('models') or k.startswith('utils')}
        for key in list(sys.modules):
            if key == 'datasets' or key == 'eval' or key.startswith('models') or key.startswith('utils'):
                del sys.modules[key]
        self._old_sys_path = list(sys.path)
        sys.path.insert(0, str(self.repo))
        return self

    def __exit__(self, exc_type, exc, tb):
        for key in list(sys.modules):
            if key == 'datasets' or key == 'eval' or key.startswith('models') or key.startswith('utils'):
                del sys.modules[key]
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_sys_path


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            tokens = self.model.patchembed(inputs)
        elif isinstance(inputs, (list, tuple)):
            # DCT group inputs are usually dict inside a default-collated mapping; old DCT path may be tuple.
            try:
                tokens = self.model.patchembed(*inputs)
            except TypeError:
                tokens = self.model.patchembed(inputs)
        else:
            tokens = self.model.patchembed(inputs)
        encoded = self.model.encoder(tokens)
        normalized = self.model.classhead.ch_lrnorm(encoded)
        return normalized.mean(dim=1)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if state and next(iter(state)).startswith('module.'):
        state = {k.removeprefix('module.'): v for k, v in state.items()}
    return state


def build_stagewise_model_and_loaders(spec: StageSpec, split: str, batch_size: int, num_workers: int, prefetch_factor: int, in_order: bool):
    import datasets as ds
    from models.plainvit import ViT

    dataset_name = ds.stage_to_dataset(spec.stage)
    index = STAGEWISE_REPO / ('assets/indexbase_train.csv' if split == 'train' else 'assets/indexbase_val.csv')
    dataset_type = 'val' if split == 'train' else 'test'
    dataset_obj = ds.imagenet_dataset_indexing(
        indexfile=str(index), type=dataset_type,
        basepath=str(DATA_ROOT), load_mode=spec.stage, stage=spec.stage,
        dtype=torch.float32, rlc_lmax=spec.rlc_lmax,
        ops_list=None, num_ops=2, ops_magnitude=3,
        freqnorm=spec.freqnorm, freqnorm_stats=str(spec.freqnorm_stats) if spec.freqnorm_stats else None,
    )
    # Keep direct event stages on the same requantized coefficient path used in training.
    # `dataset_type='val'` removes random crop/flip/RandAugment while preserving dequant->requant semantics.
    loader = DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        collate_fn=ds.optimized_stage_collate_fn,
        in_order=in_order,
    )
    model = ViT(
        in_channels=3, patch_size=16, emb_size=384, depth=12, n_classes=1000,
        drop_p=0.0, pixel_space=spec.stage, stage=spec.stage, ver=1,
        use_subblock=False, numeric_feature_dim=spec.numeric_feature_dim or 384,
        rlc_lmax=spec.rlc_lmax, device='cpu', dtype=torch.float32,
        num_heads=6, head_size=64,
    )
    model.load_state_dict(load_state_dict(spec.checkpoint, torch.device('cpu')))
    return model, loader, dataset_name


def build_old_model_and_loaders(spec: StageSpec, split: str, batch_size: int, num_workers: int, prefetch_factor: int):
    with RepoContext(OLD_REPO):
        import datasets as ds
        from models.plainvit import ViT

        index = OLD_REPO / ('assets/indexbase_train.csv' if split == 'train' else 'assets/indexbase_val.csv')
        dataset_name = 'imagenet' if spec.stage == 'rgb' else 'imagenet_mcu_bytes'
        if spec.stage == 'rgb':
            # For probe, train features intentionally use deterministic val/test transform.
            dataset_obj = ds.imagenet_dataset_indexing(
                indexfile=str(index), type='test', basepath=str(DATA_ROOT), load_mode='RGB', dtype=torch.float32,
            )
            transform = ds.get_transform(dataset='imagenet', type='test', dtype=torch.float32)
            dataset_obj = ds.SubsetWithTransform(dataset_obj, dataset='imagenet', transform=transform)
            collate_fn = None
            pixel_space = 'RGB'
            ver = 1
        else:
            dataset_obj = ds.imagenet_dataset_indexing(
                indexfile=str(index), type='test' if split != 'train' else 'test',
                basepath=str(DATA_ROOT), load_mode='MCU_BYTES', dtype=torch.float32,
            )
            collate_fn = ds.optimized_mcu_collate_fn
            pixel_space = 'MCU_BYTES'
            ver = 1
        loader = DataLoader(
            dataset_obj, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
            prefetch_factor=None if num_workers == 0 else prefetch_factor,
            collate_fn=collate_fn,
        )
        model = ViT(
            in_channels=3, patch_size=16, emb_size=384, depth=12, n_classes=1000,
            drop_p=0.0, pixel_space=pixel_space, ver=ver, use_subblock=True,
            device='cpu', dtype=torch.float32, num_heads=6, head_size=64,
        )
        model.load_state_dict(load_state_dict(spec.checkpoint, torch.device('cpu')))
    return model, loader, dataset_name


def build_model_and_loader(spec: StageSpec, split: str, batch_size: int, num_workers: int, prefetch_factor: int, in_order: bool):
    if spec.backend == 'stagewise':
        return build_stagewise_model_and_loaders(spec, split, batch_size, num_workers, prefetch_factor, in_order)
    if spec.backend == 'old':
        return build_old_model_and_loaders(spec, split, batch_size, num_workers, prefetch_factor)
    raise ValueError(f'Unsupported backend: {spec.backend}')


def cache_features(args, spec: StageSpec, split: str):
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('Feature caching must target a CUDA device.')
    model, loader, _ = build_model_and_loader(spec, split, args.cache_batch_size, args.num_workers, args.prefetch_factor, args.in_order)

    output_dir = Path(args.cache_dir) / spec.name / split
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / 'meta.json'
    existing = sorted(output_dir.glob('shard_*.pt'))
    if args.skip_existing and meta_path.exists() and existing:
        print(f'[cache] skip existing {spec.name}/{split}: {output_dir}')
        return

    resume_samples = 0
    resume_batches = 0
    shard_idx = 0
    if args.resume_partial_cache and existing and not meta_path.exists():
        for path in existing:
            obj = torch.load(path, map_location='cpu')
            resume_samples += int(obj['labels'].numel())
        if resume_samples % args.cache_batch_size != 0:
            raise RuntimeError(f'Cannot resume partial cache with non-batch-aligned samples: {output_dir} samples={resume_samples}')
        resume_batches = resume_samples // args.cache_batch_size
        shard_idx = len(existing)
        print(f'[cache] resume partial {spec.name}/{split}: shards={shard_idx} samples={resume_samples} skip_batches={resume_batches}')
    else:
        for old in existing:
            old.unlink()
        if meta_path.exists():
            meta_path.unlink()

    if resume_batches:
        dataset = loader.dataset
        if resume_samples >= len(dataset):
            raise RuntimeError(f'Partial cache already covers dataset but meta is missing: {output_dir}')
        remaining_indices = range(resume_samples, len(dataset))
        loader = DataLoader(
            Subset(dataset, remaining_indices),
            batch_size=args.cache_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
            collate_fn=loader.collate_fn,
            in_order=args.in_order,
        )

    shard_features = []
    shard_labels = []
    shard_start = resume_samples
    sample_count = resume_samples
    started = time.time()
    amp_dtype = torch.bfloat16 if args.ampdtype == 'bf16' else torch.float16

    # Important: start DataLoader workers before creating a CUDA context.
    # DCT/JPEG workers can deadlock with fork-after-CUDA on HPC clusters.
    loader_iter = iter(loader)
    if not torch.cuda.is_available():
        raise RuntimeError('Feature caching must run on a CUDA GPU node.')
    extractor = FeatureExtractor(model).to(device)
    extractor.eval()

    with torch.inference_mode():
        pbar = tqdm(loader_iter, total=len(loader), desc=f'cache {spec.name}/{split}', dynamic_ncols=True)
        for batch_idx, (inputs, labels) in enumerate(pbar, start=1):
            inputs = move_to_device(inputs, device)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
                features = extractor(inputs)
            features = features.detach().to('cpu', dtype=torch.float16 if args.feature_dtype == 'fp16' else torch.float32)
            labels_cpu = labels.detach().to('cpu', dtype=torch.long)
            shard_features.append(features)
            shard_labels.append(labels_cpu)
            sample_count += labels_cpu.numel()
            if sample_count - shard_start >= args.shard_size:
                shard_path = output_dir / f'shard_{shard_idx:05d}.pt'
                torch.save({'features': torch.cat(shard_features, dim=0), 'labels': torch.cat(shard_labels, dim=0)}, shard_path)
                print(f'[cache] wrote {shard_path} samples={sample_count - shard_start}')
                shard_features.clear()
                shard_labels.clear()
                shard_start = sample_count
                shard_idx += 1
            if args.max_cache_batches and batch_idx >= args.max_cache_batches:
                break
    if shard_features:
        shard_path = output_dir / f'shard_{shard_idx:05d}.pt'
        torch.save({'features': torch.cat(shard_features, dim=0), 'labels': torch.cat(shard_labels, dim=0)}, shard_path)
        print(f'[cache] wrote {shard_path} samples={sample_count - shard_start}')
        shard_idx += 1

    meta = {
        'stage': spec.name, 'split': split, 'samples': sample_count,
        'feature_dim': 384, 'feature_dtype': args.feature_dtype,
        'shards': shard_idx, 'elapsed_sec': time.time() - started,
        'checkpoint': str(spec.checkpoint), 'backend': spec.backend,
    }
    with open(meta_path, 'w') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f'[cache] done {spec.name}/{split}: {json.dumps(meta, sort_keys=True)}')


def load_feature_tensor(cache_dir: Path, stage: str, split: str):
    root = cache_dir / stage / split
    shards = sorted(root.glob('shard_*.pt'))
    if not shards:
        raise FileNotFoundError(f'No shards found: {root}')
    features = []
    labels = []
    for path in shards:
        obj = torch.load(path, map_location='cpu')
        features.append(obj['features'])
        labels.append(obj['labels'])
    return torch.cat(features, dim=0).float(), torch.cat(labels, dim=0).long()


def evaluate_probe(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
            pred = logits.topk(5, dim=1).indices.t()
            matches = pred.eq(labels.view(1, -1).expand_as(pred))
            correct1 += int(matches[:1].reshape(-1).sum().item())
            correct5 += int(matches[:5].reshape(-1).sum().item())
            total += labels.numel()
    return {'loss': loss_sum / total, 'top1': correct1 / total, 'top5': correct5 / total, 'samples': total}


def train_probe(args, spec: StageSpec):
    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != 'cuda':
        raise RuntimeError('Probe training must run on a CUDA GPU node.')
    cache_dir = Path(args.cache_dir)
    train_features, train_labels = load_feature_tensor(cache_dir, spec.name, 'train')
    val_features, val_labels = load_feature_tensor(cache_dir, spec.name, 'val')
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.probe_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=args.probe_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.Linear(384, 1000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = {'top1': -1.0, 'top5': -1.0, 'epoch': 0, 'loss': math.inf}
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total += labels.numel()
        scheduler.step()
        metrics = evaluate_probe(model, val_loader, device)
        row = {
            'epoch': epoch, 'train_loss': total_loss / total,
            'val_loss': metrics['loss'], 'val_top1': metrics['top1'], 'val_top5': metrics['top5'],
            'lr': scheduler.get_last_lr()[0],
        }
        history.append(row)
        if metrics['top1'] > best['top1']:
            best = {'top1': metrics['top1'], 'top5': metrics['top5'], 'epoch': epoch, 'loss': metrics['loss']}
        print(f"[probe] {spec.name} epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_top1={100*metrics['top1']:.3f} val_top5={100*metrics['top5']:.3f} best={100*best['top1']:.3f}@{best['epoch']}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / f'{spec.name}_history.csv'
    with open(history_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    result = {
        'stage': spec.name, 'backend': spec.backend, 'stage_name': spec.stage,
        'checkpoint': str(spec.checkpoint), 'e2e_top1': spec.e2e_top1, 'e2e_top5': spec.e2e_top5,
        'probe_best_epoch': best['epoch'], 'probe_top1': 100 * best['top1'], 'probe_top5': 100 * best['top5'],
        'probe_loss': best['loss'], 'epochs': args.epochs, 'lr': args.lr,
        'elapsed_sec': time.time() - started, 'history': str(history_path),
    }
    result_path = out_dir / f'{spec.name}_result.json'
    with open(result_path, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f'[probe] wrote {result_path}: {json.dumps(result, sort_keys=True)}')
    return result


def aggregate_results(args):
    out_dir = Path(args.output_dir)
    rows = []
    for name in args.stages:
        path = out_dir / f'{name}_result.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    csv_path = out_dir / 'probe_results.csv'
    if rows:
        keys = ['stage','stage_name','backend','e2e_top1','e2e_top5','probe_top1','probe_top5','probe_best_epoch','probe_loss','checkpoint']
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in keys})
    md_path = Path(args.markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# Linear Probe Results\n')
    lines.append(f'Updated: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}\n')
    lines.append('Feature point: frozen backbone output after `classhead.ch_lrnorm` and mean-pool, before the original MLP head. Probe head is a single `Linear(384,1000)` trained with `weight_decay=0`.\n')
    lines.append('| Stage | Exact stage/domain | Backend | E2E Top-1 | E2E Top-5 | Probe Top-1 | Probe Top-5 | Best epoch | Checkpoint |\n')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---|\n')
    for row in rows:
        e2e5 = '' if row.get('e2e_top5') is None else f"{row['e2e_top5']:.3f}"
        lines.append(
            f"| {row['stage']} | `{row['stage_name']}` | {row['backend']} | {row['e2e_top1']:.3f} | {e2e5} | {row['probe_top1']:.3f} | {row['probe_top5']:.3f} | {row['probe_best_epoch']} | `{row['checkpoint']}` |\n"
        )
    lines.append('\n## Claim Boundary\n')
    lines.append('- Linear probe accuracy measures linear separability of the frozen representation, not information sufficiency.\n')
    lines.append('- `rgb` and `raw_byte` use the older `/path/to/DDViT` checkpoints; DCT/RLC/Huffman rows use `/path/to/DDViT_stagewise_jpeg`.\n')
    lines.append(f'\nRaw CSV: `{csv_path}`\n')
    md_path.write_text(''.join(lines))
    print(f'[aggregate] wrote {csv_path}')
    print(f'[aggregate] wrote {md_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cache','train','aggregate','all'], required=True)
    parser.add_argument('--stages', nargs='+', default=['rgb','dct_float_freqnorm','quant_dct_freqnorm','rlc_conv','huff_conv','raw_byte'], choices=sorted(STAGE_SPECS))
    parser.add_argument('--splits', nargs='+', default=['train','val'], choices=['train','val'])
    parser.add_argument('--cache_dir', default=str(STAGEWISE_REPO / 'outputs/linear_probe_features'))
    parser.add_argument('--output_dir', default=str(STAGEWISE_REPO / 'outputs/linear_probe_results'))
    parser.add_argument('--markdown', default=str(STAGEWISE_REPO / 'LINEAR_PROBE_RESULTS.md'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cache_batch_size', type=int, default=1024)
    parser.add_argument('--probe_batch_size', type=int, default=8192)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=1)
    parser.add_argument('--in_order', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--feature_dtype', choices=['fp16','fp32'], default='fp16')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ampdtype', choices=['fp16','bf16'], default='bf16')
    parser.add_argument('--shard_size', type=int, default=50000)
    parser.add_argument('--max_cache_batches', type=int, default=0, help='Smoke/debug only; 0 means full split')
    parser.add_argument('--cache_start_method', choices=['none','fork','spawn','forkserver'], default='fork')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--resume_partial_cache', action='store_true', help='Resume an incomplete split from already written full-batch shards.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache_start_method != 'none':
        try:
            mp.set_start_method(args.cache_start_method, force=True)
        except RuntimeError:
            pass
    for name in args.stages:
        spec = STAGE_SPECS[name]
        if not spec.checkpoint.exists():
            raise FileNotFoundError(spec.checkpoint)
    if args.mode in ('cache', 'all'):
        for name in args.stages:
            for split in args.splits:
                cache_features(args, STAGE_SPECS[name], split)
    if args.mode in ('train', 'all'):
        for name in args.stages:
            train_probe(args, STAGE_SPECS[name])
    if args.mode in ('aggregate', 'all'):
        aggregate_results(args)


if __name__ == '__main__':
    main()
