"""Helpers for latency and size measurements (pure Python; the timing loop lives in ``scripts/measure_latency.py``)."""
import os
from typing import Dict, Sequence

import numpy as np


def summarize_times(seconds: Sequence[float]) -> Dict[str, float]:
    """Mean, median, 95th percentile, min and max of per-receipt times, in milliseconds."""
    if len(seconds) == 0:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "n": int(len(ms)),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
    }


def dir_size_mb(path: str) -> float:
    """Total size of a file or of every file below a folder, in megabytes (10^6 bytes)."""
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / 1e6


def mean_stage_seconds(timings: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of every stage found in the ``timing_sec`` dictionaries of the OCR JSON files."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for t in timings:
        for key, value in (t or {}).items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def format_latency_table(rows: Sequence[Dict[str, object]]) -> str:
    """Markdown table of the measurement rows written by ``measure_latency.py``."""
    head = ("| System | Device | Receipts | Mean ms | Median ms | p95 ms | Parameters | Size MB |\n"
            "|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        lines.append(f"| {r['system']} | {r['device']} | {r['n']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | "
                     f"{r['p95_ms']:.1f} | {r['parameters']:,} | {r['size_mb']:.0f} |")
    return head + "\n".join(lines) + "\n"
