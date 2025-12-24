from __future__ import annotations

import os
import numpy as np
import pandas as pd

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]

INPUT_FILES = [
    "results/output/adult_income_test_summary.csv",
    "results/output/bank_marketing_test_summary.csv",
    "results/output/breast_cancer_test_summary.csv",
    "results/output/diabetese_test_summary.csv",
    "results/output/in_vehicle_coupon_test_summary.csv",
]


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x)


def format_mean_std(mean, std, decimals: int = 4) -> str:
    if _is_number(mean) and _is_number(std):
        return f"{float(mean):.{decimals}f} ± {float(std):.{decimals}f}"
    if _is_number(mean) and not _is_number(std):
        return f"{float(mean):.{decimals}f} ± NA"
    return "NA ± NA"


def ensure_metric_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Ensures df has METRICS columns as formatted 'mean ± std' strings.
    Uses existing formatted columns if present; otherwise builds from *_mean/*_std.
    """
    for m in METRICS:
        if m in df.columns:
            # Already formatted (likely "mean ± std")
            continue

        mean_col = f"{m}_mean"
        std_col = f"{m}_std"

        if mean_col in df.columns or std_col in df.columns:
            means = df[mean_col] if mean_col in df.columns else np.nan
            stds = df[std_col] if std_col in df.columns else np.nan
            df[m] = [format_mean_std(means.iloc[i], stds.iloc[i], decimals) for i in range(len(df))]
        else:
            df[m] = "NA ± NA"

    return df


def infer_dataset_name(path: str) -> str:
    # e.g. breast_cancer_test_summary.csv -> breast_cancer
    base = os.path.basename(path)
    return base.replace("_test_summary.csv", "")


def load_one(path: str, decimals: int = 4) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure dataset + model exist
    if "dataset" not in df.columns:
        df["dataset"] = infer_dataset_name(path)

    if "model" not in df.columns:
        raise ValueError(f"'model' column not found in {path}. Columns: {list(df.columns)}")

    df = ensure_metric_columns(df, decimals=decimals)

    # Keep only requested columns
    out = df[["dataset", "model"] + METRICS].copy()
    return out


def build_big_table(files: list[str], decimals: int = 4) -> pd.DataFrame:
    parts = [load_one(p, decimals=decimals) for p in files]
    big = pd.concat(parts, ignore_index=True)

    # Nice ordering
    big = big.sort_values(["dataset", "model"], na_position="last").reset_index(drop=True)
    return big


if __name__ == "__main__":
    big_df = build_big_table(INPUT_FILES, decimals=4)

    out_path = "results/output/all_datasets_big_test_table.csv"
    big_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(big_df.to_markdown(index=False))
