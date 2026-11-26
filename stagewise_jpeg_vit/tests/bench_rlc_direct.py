import argparse
import os
import sys
import time
import gc

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import datasets as ds


DC_ONLY_STAGES = {
    "rlc_dc", "rlc_dc_simple", "rlc_dc_attnpool",
    "rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat", "rlc_dc_2d_cp",
    "rlc_dc_2d_direct", "rlc_dc_2d_max_direct",
    "rlc_dc_flat_direct", "rlc_dc_2d_cp_direct",
}


def _bench_one(args, stage):
    stage = ds.normalize_stage_name(stage)
    lmax = 8 if stage in DC_ONLY_STAGES else args.rlc_lmax
    dataset = ds.imagenet_dataset_indexing(
        indexfile=args.index_path,
        basepath=args.basepath,
        load_mode=stage,
        stage=stage,
        type="train",
        rlc_lmax=lmax,
        ops_list=None,
        num_ops=args.num_ops,
        ops_magnitude=args.ops_magnitude,
        dtype=torch.float32,
    )
    if stage.endswith("_direct"):
        dataset.rlc_direct_randaug = args.rlc_direct_randaug
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": True,
        "collate_fn": ds.optimized_stage_collate_fn,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    setup_start = time.perf_counter()
    iterator = iter(loader)
    iterator_elapsed = time.perf_counter() - setup_start
    warmup_elapsed = 0.0
    for _ in range(args.warmup_batches):
        t0 = time.perf_counter()
        next(iterator)
        warmup_elapsed += time.perf_counter() - t0

    batch_times = []
    start = time.perf_counter()
    for _ in range(args.num_batches):
        t0 = time.perf_counter()
        next(iterator)
        batch_times.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start
    total_elapsed = iterator_elapsed + warmup_elapsed + elapsed

    times = torch.tensor(batch_times)
    samples = args.num_batches * args.batch_size
    print(f"stage={stage} lmax={lmax} workers={args.num_workers} batch={args.batch_size}")
    print(
        f"iterator={iterator_elapsed:.2f}s warmup={warmup_elapsed:.2f}s "
        f"timed={elapsed:.2f}s total={total_elapsed:.2f}s batches={args.num_batches}"
    )
    print(
        f"throughput_timed={samples / elapsed:.1f} samples/s "
        f"throughput_total={samples / total_elapsed:.1f} samples/s"
    )
    print(
        "batch_time "
        f"mean={times.mean().item():.3f}s "
        f"p50={times.quantile(0.50).item():.3f}s "
        f"p90={times.quantile(0.90).item():.3f}s "
        f"p99={times.quantile(0.99).item():.3f}s "
        f"max={times.max().item():.3f}s"
    )
    print(flush=True)
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    del iterator
    del loader
    del dataset
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append", default=None)
    parser.add_argument("--index_path", default=os.path.join(REPO, "assets", "indexbase_train.csv"))
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=80)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--rlc_lmax", type=int, default=320)
    parser.add_argument("--num_ops", type=int, default=2)
    parser.add_argument("--ops_magnitude", type=int, default=3)
    parser.add_argument("--no_randaugment", dest="rlc_direct_randaug", action="store_false")
    parser.set_defaults(rlc_direct_randaug=True)
    args = parser.parse_args()

    stages = args.stage or [
        "rlc_dcac_2d_cp_direct",
        "huff_dcac_2d_cp_direct",
        "rlc_dcac_2d_cp_concat_conv_direct",
        "huff_dcac_2d_cp_concat_conv_direct",
    ]
    for stage in stages:
        _bench_one(args, stage)


if __name__ == "__main__":
    main()
