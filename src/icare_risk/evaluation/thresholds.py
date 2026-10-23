"""Select operating thresholds on development data and evaluate them on held-out data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


def select_thresholds(
    y_true: Iterable,
    probabilities: Iterable,
    target_sensitivities: Iterable[float] = (0.80, 0.90),
    include_youden: bool = True,
) -> dict[str, float]:
    """Select thresholds using development data only."""
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    valid = y.notna() & p.notna()
    y = y[valid].astype(int).to_numpy()
    p = p[valid].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires both outcome classes.")

    fpr, tpr, thresholds = roc_curve(y, p)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    selected: dict[str, float] = {}

    if include_youden:
        selected["youden"] = float(thresholds[np.argmax(tpr - fpr)])

    for target in target_sensitivities:
        eligible = np.flatnonzero(tpr >= target)
        if len(eligible):
            # Among thresholds meeting the sensitivity target, prefer specificity.
            best = eligible[np.argmax(1.0 - fpr[eligible])]
        else:
            best = int(np.argmax(tpr))
        selected[f"sensitivity_{target:.2f}"] = float(thresholds[best])
    return selected


def evaluate_fixed_thresholds(
    y_true: Iterable,
    probabilities: Iterable,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Evaluate thresholds that were selected independently of this dataset."""
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    valid = y.notna() & p.notna()
    y = y[valid].astype(int).to_numpy()
    p = p[valid].to_numpy()
    rows = []
    for name, threshold in thresholds.items():
        predicted = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append({
            "threshold_name": name,
            "threshold": float(threshold),
            "n": len(y),
            "events": int(y.sum()),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "predicted_positive_rate": float(predicted.mean()),
        })
    return pd.DataFrame(rows)


def threshold_analysis(
    predictions: pd.DataFrame,
    target_column: str,
    model_columns: list[str],
    split_column: str = "split",
    development_labels: tuple[str, ...] = ("train", "validation"),
    test_label: str = "test",
    target_sensitivities: Iterable[float] = (0.80, 0.90),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select thresholds on development predictions and apply them to test predictions."""
    development = predictions[predictions[split_column].isin(development_labels)]
    test = predictions[predictions[split_column].eq(test_label)]
    selected_rows, test_rows = [], []
    for column in model_columns:
        thresholds = select_thresholds(
            development[target_column],
            development[column],
            target_sensitivities=target_sensitivities,
        )
        for name, value in thresholds.items():
            selected_rows.append({"model": column.removesuffix("_prob"), "prediction_column": column, "threshold_name": name, "threshold": value})
        evaluated = evaluate_fixed_thresholds(test[target_column], test[column], thresholds)
        evaluated.insert(0, "model", column.removesuffix("_prob"))
        evaluated.insert(1, "prediction_column", column)
        test_rows.append(evaluated)
    return pd.DataFrame(selected_rows), pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
