import argparse
import tempfile
import time
import types
from pathlib import Path

import torch

import byte_manip as bm
import dct_manip as dm
import datasets as new_ds


def load_dataset_module(module_path: Path, module_name: str):
    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    exec(module_path.read_text(), module.__dict__)
    return module


def load_old_dataset_module(repo_root: Path, old_module_path: Path | None = None):
    candidate = old_module_path or (repo_root / 'datasets.py.bak_payload')
    if not candidate.exists():
        return None, candidate
    return load_dataset_module(candidate, 'datasets_bak_payload'), candidate


def collect_sample_paths(repo_root: Path, imagenet_root: Path, limit: int):
    rows = (repo_root / 'assets' / 'indexbase_train.csv').read_text().splitlines()[1:]
    out = []
    for row in rows:
        rel = row.split(',')[0]
        p = imagenet_root / rel
        if p.exists() and p.suffix.lower() in {'.jpeg', '.jpg'}:
            out.append(p)
        if len(out) >= limit:
            break
    if len(out) < limit:
        raise RuntimeError(f'Only found {len(out)} JPEG samples, expected {limit}')
    return out


def build_eval_coefficients(sample: Path):
    _, quant, Y, cbcr = dm.read_coefficients(str(sample))
    coeffs = (Y, cbcr) if cbcr is not None else (Y,)
    transform = new_ds.get_mcu_coeff_transform(type='eval', dtype_resize=torch.float32)
    coeffs = transform(coeffs)
    Y_out = coeffs[0].short().contiguous()
    cbcr_out = coeffs[1].short().contiguous() if len(coeffs) > 1 else None
    new_dim = torch.tensor([[224, 224]], dtype=torch.int32)
    quant = quant.short().contiguous()
    return new_dim, quant, Y_out, cbcr_out


def compare_low_level_equivalence(sample: Path):
    new_dim, quant, Y_out, cbcr_out = build_eval_coefficients(sample)
    with tempfile.NamedTemporaryFile(suffix='.JPEG', delete=False) as f:
        tmp = f.name
    try:
        dm.write_coefficients(tmp, new_dim, quant, Y_out, cbcr_out)
        packed_old = bm.read_mcu_packed_bytes(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    jpeg_buffer = dm.write_coefficients_to_buffer(new_dim, quant, Y_out, cbcr_out)
    packed_new = bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)
    all_equal = (
        packed_old['header'] == packed_new['header']
        and len(packed_old['mcus']) == len(packed_new['mcus'])
        and all(
            a['payload_list'] == b['payload_list']
            and a['bit_start'] == b['bit_start']
            and a['bit_end'] == b['bit_end']
            for a, b in zip(packed_old['mcus'], packed_new['mcus'])
        )
    )
    return {
        'header_equal': packed_old['header'] == packed_new['header'],
        'mcu_count_old': len(packed_old['mcus']),
        'mcu_count_new': len(packed_new['mcus']),
        'all_payload_equal': all_equal,
    }


def benchmark_stage_only(sample: Path, rounds: int, warmup: int):
    new_dim, quant, Y_out, cbcr_out = build_eval_coefficients(sample)
    for _ in range(warmup):
        with tempfile.NamedTemporaryFile(suffix='.JPEG', delete=False) as f:
            tmp = f.name
        dm.write_coefficients(tmp, new_dim, quant, Y_out, cbcr_out)
        bm.read_mcu_packed_bytes(tmp)
        Path(tmp).unlink(missing_ok=True)
        jpeg_buffer = dm.write_coefficients_to_buffer(new_dim, quant, Y_out, cbcr_out)
        bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)

    file_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix='.JPEG', delete=False) as f:
            tmp = f.name
        dm.write_coefficients(tmp, new_dim, quant, Y_out, cbcr_out)
        bm.read_mcu_packed_bytes(tmp)
        Path(tmp).unlink(missing_ok=True)
        file_times.append(time.perf_counter() - t0)

    buffer_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        jpeg_buffer = dm.write_coefficients_to_buffer(new_dim, quant, Y_out, cbcr_out)
        bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)
        buffer_times.append(time.perf_counter() - t0)

    file_mean = sum(file_times) / len(file_times)
    buffer_mean = sum(buffer_times) / len(buffer_times)
    return {
        'rounds': rounds,
        'file_mean_ms': 1000.0 * file_mean,
        'buffer_mean_ms': 1000.0 * buffer_mean,
        'buffer_speedup_vs_file': file_mean / buffer_mean,
        'file_min_ms': 1000.0 * min(file_times),
        'buffer_min_ms': 1000.0 * min(buffer_times),
    }


def benchmark_end_to_end(repo_root: Path, imagenet_root: Path, samples: int, old_module_path: Path | None = None):
    sample_paths = collect_sample_paths(repo_root, imagenet_root, samples)

    start = time.perf_counter()
    new_outputs = [new_ds.coefficients_to_mcu_payload_tensor(str(p), type='eval', lmax=192) for p in sample_paths]
    new_total = time.perf_counter() - start

    result = {
        'samples': samples,
        'new_total_sec': new_total,
        'new_sec_per_sample': new_total / samples,
        'sample_example': str(sample_paths[0]),
    }

    old_ds, resolved_old_path = load_old_dataset_module(repo_root, old_module_path)
    if old_ds is None:
        result.update({
            'old_available': False,
            'skipped_reason': (
                f'No old dataset module found at {resolved_old_path}. '
                'Provide --old-module-path or place datasets.py.bak_payload under repo root.'
            ),
        })
        return result

    start = time.perf_counter()
    old_outputs = [old_ds.coefficients_to_mcu_inputs(str(p), type='eval', lmax=192)[0] for p in sample_paths]
    old_total = time.perf_counter() - start

    exact_equal = all(torch.equal(a.to(torch.int64), b.to(torch.int64)) for a, b in zip(old_outputs, new_outputs))
    result.update({
        'old_available': True,
        'old_module_path': str(resolved_old_path),
        'old_total_sec': old_total,
        'old_sec_per_sample': old_total / samples,
        'buffer_speedup_vs_file': old_total / new_total,
        'exact_tensor_equal': exact_equal,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--imagenet-root', type=Path, default=Path('/path/to/imagenet'))
    parser.add_argument('--old-module-path', type=Path, default=None)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()

    sample_paths = collect_sample_paths(args.repo_root, args.imagenet_root, max(1, args.samples))
    stage_result = benchmark_stage_only(sample_paths[0], args.rounds, args.warmup)
    eq_result = compare_low_level_equivalence(sample_paths[0])
    e2e_result = benchmark_end_to_end(args.repo_root, args.imagenet_root, args.samples, old_module_path=args.old_module_path)

    print({'equivalence': eq_result})
    print({'stage_only': stage_result})
    print({'end_to_end_eval': e2e_result})


if __name__ == '__main__':
    main()
