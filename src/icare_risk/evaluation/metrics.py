from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

def encode_binary_target(values, target_mapping: Mapping | None = None) -> pd.Series:
    mapping = target_mapping or {}
    positive = {str(v).strip().lower() for v in mapping.get(
        "positive_values", [1, "1", "positive", "esbl", "yes", True]
    )}
    negative = {str(v).strip().lower() for v in mapping.get(
        "negative_values", [0, "0", "negative", "non-esbl", "non esbl", "no", False]
    )}
    raw = pd.Series(values)
    normalised = raw.map(lambda x: str(x).strip().lower() if pd.notna(x) else None)
    result = pd.Series(np.nan, index=raw.index, dtype="float64")
    result[normalised.isin(negative)] = 0
    result[normalised.isin(positive)] = 1
    return result

def _valid_binary_data(y_true, y_prob, target_mapping=None):
    y = encode_binary_target(y_true, target_mapping)
    p = pd.to_numeric(pd.Series(y_prob, index=y.index), errors="coerce")
    valid = y.notna() & p.notna() & p.between(0, 1)
    return y.loc[valid].astype(int).to_numpy(), p.loc[valid].to_numpy()

def _empty_threshold(threshold):
    return {k: float(threshold) if k == "threshold" else np.nan for k in (
        "threshold", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
        "ppv", "npv", "accuracy", "balanced_accuracy", "f1"
    )}

def _threshold_metrics(y, p, threshold):
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "ppv": float(precision_score(y, predicted, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }

def compute_metrics(y_true, y_prob, *, threshold=0.5, target_mapping=None) -> dict:
    """Compute discrimination, calibration proxy, and threshold metrics."""
    y, p = _valid_binary_data(y_true, y_prob, target_mapping)
    result = {
        "n": len(y),
        "n_events": int(y.sum()) if len(y) else 0,
        "event_rate": float(y.mean()) if len(y) else np.nan,
    }
    if len(y) == 0 or np.unique(y).size < 2:
        return {
            **result, "auroc": np.nan, "auprc": np.nan,
            "brier_score": np.nan, **_empty_threshold(threshold),
        }
    result.update({
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    })
    return {**result, **_threshold_metrics(y, p, threshold)}

def evaluate_prediction_table(
    df: pd.DataFrame, *, target_column: str,
    prediction_columns: Sequence[str] | None = None, subset=None,
    threshold=0.5, target_mapping=None,
) -> pd.DataFrame:
    """Evaluate all probability columns in a common prediction table."""
    work = df.loc[subset].copy() if subset is not None else df.copy()
    columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    rows = []
    for column in columns:
        rows.append({
            "model": column.removesuffix("_prob"),
            "prediction_column": column,
            **compute_metrics(
                work[target_column], work[column],
                threshold=threshold, target_mapping=target_mapping,
            ),
        })
    return pd.DataFrame(rows)
