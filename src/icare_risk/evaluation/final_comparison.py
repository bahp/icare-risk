"""Utilities for consolidating manuscript-ready model comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import calibration_summary
from .metrics import compute_metrics


def evaluate_models_on_table(
    df: pd.DataFrame,
    *,
    target_column: str,
    prediction_columns: Sequence[str] | None = None,
    subset: pd.Series | np.ndarray | None = None,
    target_mapping: Mapping | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a unified point-estimate comparison table from predictions."""
    work = df.loc[subset].copy() if subset is not None else df.copy()
    columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    rows = []
    for column in columns:
        model = column.removesuffix("_prob")
        metrics = compute_metrics(
            work[target_column], work[column], threshold=threshold,
            target_mapping=target_mapping,
        )
        calibration = calibration_summary(
            work[target_column], work[column], target_mapping=target_mapping,
        )
        rows.append({
            "model": model,
            "prediction_column": column,
            **metrics,
            "calibration_intercept": calibration["calibration_intercept"],
            "calibration_slope": calibration["calibration_slope"],
            "observed_expected_ratio": calibration["observed_expected_ratio"],
        })
    return pd.DataFrame(rows)


def add_feature_metadata(
    comparison: pd.DataFrame,
    *,
    model_configs: Sequence[Mapping] | None = None,
    feature_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Attach configured feature-set and feature-count metadata."""
    result = comparison.copy()
    config_map = {c["name"]: c for c in (model_configs or [])}
    result["feature_set"] = result["model"].map(
        lambda name: config_map.get(name, {}).get("feature_set", pd.NA)
    )
    result["model_type"] = result["model"].map(
        lambda name: config_map.get(name, {}).get("type", pd.NA)
    )
    result["n_features"] = result["model"].map(
        lambda name: (feature_counts or {}).get(name, pd.NA)
    )
    return result


def rank_models(
    comparison: pd.DataFrame,
    *,
    primary_metric: str = "auroc",
    secondary_metrics: Sequence[str] = ("auprc", "brier_score"),
) -> pd.DataFrame:
    """Rank models, with higher discrimination and lower error preferred."""
    result = comparison.copy()
    if primary_metric not in result.columns:
        raise KeyError(f"Metric not found: {primary_metric}")
    result["primary_rank"] = result[primary_metric].rank(
        ascending=False, method="min", na_option="bottom"
    )
    score = result[primary_metric].rank(ascending=False, pct=True, na_option="keep")
    if "auprc" in result:
        score = score.fillna(0) + result["auprc"].rank(ascending=False, pct=True, na_option="keep").fillna(0)
    if "brier_score" in result:
        score = score + (1 - result["brier_score"].rank(ascending=True, pct=True, na_option="keep").fillna(1))
    result["composite_score"] = score
    return result.sort_values(
        ["composite_score", primary_metric], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


def consolidate_result_files(
    directory: str | Path,
    *,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and concatenate compatible saved result tables."""
    paths = sorted(Path(directory).glob(pattern))
    tables = []
    for path in paths:
        try:
            table = pd.read_csv(path)
        except Exception:
            continue
        if not table.empty:
            table.insert(0, "source_file", path.name)
            tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()


def save_final_comparison(
    comparison: pd.DataFrame,
    output_directory: str | Path,
    *,
    filename: str = "final_model_comparison.csv",
) -> Path:
    """Save a comparison table and return its path."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    comparison.to_csv(path, index=False)
    return path
