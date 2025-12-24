#!/usr/bin/env python3
"""
Build a one-row-per-model summary table from *_results.json files.

- Reads JSON files (default: /mnt/data/*_results.json)
- Extracts test_metrics.mean and test_metrics.std
- Produces a wide table:
    model | dataset | <metric> (mean±std) | <metric>_mean | <metric>_std | ...
- Saves CSV and prints a markdown preview
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Tuple

import pandas as pd


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def format_mean_std(mean: Any, std: Any, decimals: int = 4) -> str:
    """Format as 'mean ± std' with fixed decimals when numeric, else fallback."""
    if _is_number(mean) and _is_number(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    if _is_number(mean) and not _is_number(std):
        return f"{mean:.{decimals}f} ± NA"
    return f"{mean} ± {std}"


def load_result_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_test_metrics(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    test = doc.get("test_metrics", {}) or {}
    mean = test.get("mean", {}) or {}
    std = test.get("std", {}) or {}
    if not isinstance(mean, dict) or not isinstance(std, dict):
        raise ValueError("test_metrics.mean and test_metrics.std must both be dicts.")
    return mean, std


def build_summary(
    json_paths: List[str],
    decimals: int = 4,
    keep_numeric: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    all_metrics: set[str] = set()

    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for p in json_paths:
        doc = load_result_json(p)
        loaded.append((p, doc))
        mean, std = extract_test_metrics(doc)
        all_metrics |= set(mean.keys()) | set(std.keys())

    # Stable column order (you can customize)
    preferred_order = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"
    ]
    metrics_sorted = [m for m in preferred_order if m in all_metrics] + sorted(
        [m for m in all_metrics if m not in preferred_order]
    )

    for path, doc in loaded:
        mean, std = extract_test_metrics(doc)

        row: Dict[str, Any] = {
            "model": doc.get("model", os.path.basename(path)),
            "dataset": doc.get("dataset", None),
            "experiment_name": doc.get("experiment_name", None),
            "timestamp": doc.get("timestamp", None),
            "source_file": os.path.basename(path),
        }

        for m in metrics_sorted:
            m_mean = mean.get(m, float("nan"))
            m_std = std.get(m, float("nan"))
            row[m] = format_mean_std(m_mean, m_std, decimals=decimals)

            if keep_numeric:
                row[f"{m}_mean"] = m_mean
                row[f"{m}_std"] = m_std

        rows.append(row)

    df = pd.DataFrame(rows)

    # Put identity columns first
    id_cols = ["model", "dataset", "experiment_name", "timestamp", "source_file"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + metric_cols]

    # Nice sorting
    df = df.sort_values(["dataset", "model"], na_position="last").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="/mnt/data/*_results.json",
        help="Glob pattern for result JSON files",
    )
    ap.add_argument(
        "--decimals", type=int, default=4, help="Decimals for mean/std formatting"
    )
    ap.add_argument(
        "--no-numeric",
        action="store_true",
        help="If set, do not include <metric>_mean and <metric>_std numeric columns",
    )
    ap.add_argument(
        "--out",
        default="/mnt/data/test_metrics_summary.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched glob: {args.glob}")

    df = build_summary(
        json_paths=paths,
        decimals=args.decimals,
        keep_numeric=not args.no_numeric,
    )

    df.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
