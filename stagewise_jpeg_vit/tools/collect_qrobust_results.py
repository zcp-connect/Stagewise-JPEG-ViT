#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TEST_RE = re.compile(r"Test Acc:\s*([0-9.]+)\s*%")
LOG_RE = re.compile(r"(?P<stage>.+)__(?P<quality>orig|q\d+)\.log$")
TB_RE = re.compile(
    r"(?P<stage>.+)__(?P<quality>orig|q\d+)_lr[0-9.Ee+-]+_wd[0-9.Ee+-]+_drop[0-9.]+$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/qrobust")
    parser.add_argument("--tb-root", default="outputs/qrobust_eval/tensorboard")
    parser.add_argument("--out", default="outputs/qrobust_results.csv")
    return parser.parse_args()


def read_tb_acc(test_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError:
        return None

    event_files = sorted(test_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return None
    best = None
    for _event_file in event_files:
        accumulator = EventAccumulator(str(test_dir), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception:
            continue
        if "Acc/Test" not in accumulator.Tags().get("scalars", []):
            continue
        values = accumulator.Scalars("Acc/Test")
        if values:
            best = values[-1].value
    return best


def collect_tb_results(tb_root):
    results = {}
    if not tb_root.exists():
        return results
    for path in sorted(p for p in tb_root.iterdir() if p.is_dir()):
        match = TB_RE.match(path.name)
        if not match:
            continue
        acc = read_tb_acc(path)
        if acc is None:
            continue
        key = (match.group("stage"), match.group("quality"))
        previous = results.get(key)
        if previous is None or path.stat().st_mtime >= previous["mtime"]:
            results[key] = {"top1": acc, "source": str(path), "mtime": path.stat().st_mtime}
    return results


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    tb_results = collect_tb_results(Path(args.tb_root))
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        match = LOG_RE.match(path.name)
        if not match:
            continue
        text = path.read_text(errors="replace")
        acc_matches = TEST_RE.findall(text)
        stage = match.group("stage")
        quality = match.group("quality")
        tb_entry = tb_results.get((stage, quality))
        if tb_entry is not None:
            top1 = f"{tb_entry['top1']:.6f}"
            status = "ok_tb"
            source = tb_entry["source"]
        elif acc_matches:
            top1 = acc_matches[-1]
            status = "ok_log"
            source = str(path)
        else:
            top1 = ""
            status = "missing"
            source = str(path)
        rows.append({
            "stage": stage,
            "quality": quality,
            "top1": top1,
            "status": status,
            "log": str(path),
            "source": source,
        })

    logged = {(row["stage"], row["quality"]) for row in rows}
    for (stage, quality), entry in sorted(tb_results.items()):
        if (stage, quality) in logged:
            continue
        rows.append({
            "stage": stage,
            "quality": quality,
            "top1": f"{entry['top1']:.6f}",
            "status": "ok_tb",
            "log": "",
            "source": entry["source"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "quality", "top1", "status", "log", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
