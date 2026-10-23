from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .metrics import _valid_binary_data

def compare_models(metrics: pd.DataFrame, *, sort_by="auroc", ascending=False) -> pd.DataFrame:
    """Sort and rank a model-metrics table."""
    required = {"model", sort_by}
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    return metrics.sort_values(sort_by, ascending=ascending, kind="stable").reset_index(drop=True)

def paired_bootstrap_auc_difference(
    y_true, prediction_a, prediction_b, *, n_bootstraps=2000, ci=0.95,
    target_mapping=None, random_state=42,
) -> dict:
    """Paired bootstrap difference in AUROC using identical resamples."""
    y_a, a = _valid_binary_data(y_true, prediction_a, target_mapping)
    y_b, b = _valid_binary_data(y_true, prediction_b, target_mapping)
    if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b):
        raise ValueError("Both prediction arrays must have the same valid observations.")
    if len(y_a) == 0 or np.unique(y_a).size < 2:
        return {
            "auroc_a": np.nan, "auroc_b": np.nan, "difference": np.nan,
            "difference_ci_lower": np.nan, "difference_ci_upper": np.nan,
        }
    point_a = roc_auc_score(y_a, a)
    point_b = roc_auc_score(y_b, b)
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_a), len(y_a))
        if np.unique(y_a[idx]).size < 2:
            continue
        differences.append(
            roc_auc_score(y_a[idx], a[idx]) -
            roc_auc_score(y_a[idx], b[idx])
        )
    alpha = (1 - ci) / 2
    return {
        "auroc_a": float(point_a), "auroc_b": float(point_b),
        "difference": float(point_a - point_b),
        "difference_ci_lower": float(np.quantile(differences, alpha)) if differences else np.nan,
        "difference_ci_upper": float(np.quantile(differences, 1 - alpha)) if differences else np.nan,
        "n_bootstraps_valid": len(differences),
    }
