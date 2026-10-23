from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from .metrics import _valid_binary_data

def calibration_summary(
    y_true, y_prob, *, target_mapping=None,
) -> dict:
    """Calculate calibration intercept, slope, Brier score, and O:E ratio."""
    y, p = _valid_binary_data(y_true, y_prob, target_mapping)
    result = {
        "n": len(y), "n_events": int(y.sum()) if len(y) else 0,
        "brier_score": np.nan, "calibration_intercept": np.nan,
        "calibration_slope": np.nan, "observed_expected_ratio": np.nan,
    }
    if len(y) == 0:
        return result
    result["brier_score"] = float(brier_score_loss(y, p))
    expected = p.sum()
    result["observed_expected_ratio"] = float(y.sum() / expected) if expected > 0 else np.nan
    if np.unique(y).size < 2:
        return result
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y)
    result["calibration_intercept"] = float(model.intercept_[0])
    result["calibration_slope"] = float(model.coef_[0, 0])
    return result

def calibration_table(
    y_true, y_prob, *, n_bins=10, strategy="quantile", target_mapping=None,
) -> pd.DataFrame:
    """Return observed and predicted risk by calibration bin."""
    y, p = _valid_binary_data(y_true, y_prob, target_mapping)
    if len(y) == 0:
        return pd.DataFrame(columns=["bin", "n", "observed_risk", "mean_predicted_risk"])
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
        bins = pd.cut(p, bins=edges, include_lowest=True, labels=False)
    else:
        q = min(n_bins, len(np.unique(p)))
        bins = pd.qcut(p, q=q, duplicates="drop", labels=False)
    counts = pd.Series(bins).value_counts(sort=False).to_numpy()
    return pd.DataFrame({
        "bin": np.arange(len(observed)) + 1,
        "n": counts[:len(observed)],
        "observed_risk": observed,
        "mean_predicted_risk": predicted,
    })

def evaluate_calibration_table(
    df: pd.DataFrame, *, target_column: str, prediction_columns=None,
    subset=None, n_bins=10, strategy="quantile", target_mapping=None,
) -> pd.DataFrame:
    work = df.loc[subset] if subset is not None else df
    columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    rows = []
    for column in columns:
        rows.append({
            "model": column.removesuffix("_prob"),
            "prediction_column": column,
            **calibration_summary(
                work[target_column], work[column],
                target_mapping=target_mapping,
            ),
        })
    return pd.DataFrame(rows)
