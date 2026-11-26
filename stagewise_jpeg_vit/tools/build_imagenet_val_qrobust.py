#!/usr/bin/env python3
import argparse
import io
import json
import multiprocessing as mp
import os
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


Q_LEVELS = ("orig", "q100", "q95", "q75", "q50", "q30", "q10")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ImageNet val JPEG quality robustness sets from imagenet-val.zip."
    )
    parser.add_argument("--zip", default="/path/to/imagenet-val.zip")
    parser.add_argument("--out", default="/path/to/imagenet1k_qrobust")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--levels", default=",".join(Q_LEVELS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_image_names(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            name for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg")) and "/" in name
        ]
    names.sort()
    return names


def save_jpeg(image, output_path, quality):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output_path,
        format="JPEG",
        quality=int(quality),
        subsampling=2,
        progressive=False,
        optimize=False,
    )


def process_one(task):
    zip_path, output_root, image_name, levels, size, overwrite = task
    rel = Path("val") / image_name
    outputs = {}
    for level in levels:
        out_path = output_root / level / rel
        outputs[level] = out_path
    if not overwrite and all(path.exists() for path in outputs.values()):
        return "skip"

    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(image_name)
    with Image.open(io.BytesIO(data)) as img:
        image = img.convert("RGB").resize((size, size), resample=Image.Resampling.BILINEAR)
        for level, out_path in outputs.items():
            if (not overwrite) and out_path.exists():
                continue
            quality = 75 if level == "orig" else int(level[1:])
            save_jpeg(image, out_path, quality)
    return "ok"


def inspect_outputs(output_root, levels, image_names):
    report = {}
    samples = image_names[:8]
    for level in levels:
        counter = Counter()
        for name in samples:
            path = output_root / level / "val" / name
            with Image.open(path) as img:
                counter[(img.format, img.mode, img.size, str(getattr(img, "layer", None)), bool(img.info.get("progressive", False)))] += 1
        report[level] = {
            "sample_count": len(samples),
            "sample_properties": [{"property": str(k), "count": v} for k, v in counter.items()],
        }
    return report


def main():
    args = parse_args()
    zip_path = Path(args.zip)
    output_root = Path(args.out)
    levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    invalid = [level for level in levels if level != "orig" and not (level.startswith("q") and level[1:].isdigit())]
    if invalid:
        raise ValueError(f"Invalid levels: {invalid}")
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    image_names = load_image_names(zip_path)
    if args.limit > 0:
        image_names = image_names[:args.limit]
    print(f"zip={zip_path}", flush=True)
    print(f"out={output_root}", flush=True)
    print(f"levels={levels}", flush=True)
    print(f"images={len(image_names)}", flush=True)

    tasks = [
        (str(zip_path), output_root, name, levels, args.size, args.overwrite)
        for name in image_names
    ]
    counts = Counter()
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, task in enumerate(tasks, 1):
            counts[process_one(task)] += 1
            if idx % 1000 == 0:
                print(f"processed={idx}/{len(tasks)} counts={dict(counts)}", flush=True)
    else:
        with mp.Pool(processes=workers) as pool:
            for idx, status in enumerate(pool.imap_unordered(process_one, tasks, chunksize=16), 1):
                counts[status] += 1
                if idx % 1000 == 0:
                    print(f"processed={idx}/{len(tasks)} counts={dict(counts)}", flush=True)

    expected = len(image_names)
    file_counts = {}
    for level in levels:
        count = sum(1 for _ in (output_root / level / "val").rglob("*.JPEG"))
        file_counts[level] = count
        if count != expected:
            raise RuntimeError(f"{level}: expected {expected} JPEG files, found {count}")

    report = {
        "source_zip": str(zip_path),
        "output_root": str(output_root),
        "levels": levels,
        "size": args.size,
        "workers": workers,
        "counts": dict(counts),
        "file_counts": file_counts,
        "jpeg_save": {
            "orig_quality": 75,
            "subsampling": 2,
            "progressive": False,
            "optimize": False,
        },
        "inspection": inspect_outputs(output_root, levels, image_names),
    }
    report_path = output_root / "qrobust_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
