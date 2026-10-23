"""Decision-curve analysis utilities for binary clinical prediction models.

All calculations operate on out-of-sample predictions. Thresholds are expressed
as predicted probabilities and net benefit is reported per patient.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_binary_target(
    values: pd.Series | Sequence,
    target_mapping: Mapping | None = None,
) -> pd.Series:
    """Convert common binary labels to nullable numeric 0/1 values."""
    series = pd.Series(values).copy()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(float)

    mapping = dict(target_mapping or {})
    positives = {str(value).strip().lower() for value in mapping.get("positive_values", [])}
    negatives = {str(value).strip().lower() for value in mapping.get("negative_values", [])}
    result = pd.Series(np.nan, index=series.index, dtype=float)
    normalised = series.astype("string").str.strip().str.lower()
    result[normalised.isin(positives)] = 1.0
    result[normalised.isin(negatives)] = 0.0
    return result


def _validate_thresholds(thresholds: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(thresholds), dtype=float)
    if values.size == 0:
        raise ValueError("At least one decision threshold is required.")
    if np.any(~np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise ValueError("Decision thresholds must be finite probabilities strictly between 0 and 1.")
    return np.unique(values)


def net_benefit(
    y_true: Sequence,
    probabilities: Sequence,
    threshold: float,
) -> float:
    """Calculate net benefit at one probability threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int).to_numpy()
    p = p.loc[valid].astype(float).to_numpy()
    if len(y) == 0:
        return np.nan
    predicted_positive = p >= threshold
    tp = np.sum(predicted_positive & (y == 1))
    fp = np.sum(predicted_positive & (y == 0))
    n = len(y)
    odds = threshold / (1.0 - threshold)
    return float((tp / n) - (fp / n) * odds)


def decision_curve(
    y_true: Sequence,
    probabilities: Sequence,
    thresholds: Iterable[float],
    model_name: str = "model",
) -> pd.DataFrame:
    """Return net benefit across thresholds for one model."""
    threshold_values = _validate_thresholds(thresholds)
    rows = [
        {
            "model": model_name,
            "threshold": threshold,
            "net_benefit": net_benefit(y_true, probabilities, threshold),
        }
        for threshold in threshold_values
    ]
    return pd.DataFrame(rows)


def evaluate_decision_curves(
    df: pd.DataFrame,
    target_column: str,
    prediction_columns: Sequence[str] | None = None,
    thresholds: Iterable[float] | None = None,
    subset: pd.Series | Sequence[bool] | None = None,
    target_mapping: Mapping | None = None,
) -> pd.DataFrame:
    """Evaluate model, treat-all, and treat-none net benefit curves.

    Parameters
    ----------
    df:
        Admission-level table containing the outcome and probability columns.
    target_column:
        Binary outcome column, or a column containing labels described by
        ``target_mapping``.
    prediction_columns:
        Probability columns. If omitted, columns ending in ``_prob`` are used.
    thresholds:
        Probability thresholds, strictly between 0 and 1. Defaults to 1%--50%.
    subset:
        Optional boolean mask, normally the untouched temporal test set.
    target_mapping:
        Optional mapping with ``positive_values`` and ``negative_values``.
    """
    work = df.copy()
    if subset is not None:
        mask = pd.Series(subset, index=work.index).fillna(False).astype(bool)
        work = work.loc[mask]
    if target_column not in work.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    prediction_columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    missing = [c for c in prediction_columns if c not in work.columns]
    if missing:
        raise KeyError(f"Prediction columns not found: {missing}")
    if not prediction_columns:
        raise ValueError("No prediction columns were supplied or detected.")

    threshold_values = _validate_thresholds(
        thresholds if thresholds is not None else np.linspace(0.01, 0.50, 100)
    )
    y = _as_binary_target(work[target_column], target_mapping)
    valid = y.notna()
    work = work.loc[valid]
    y = y.loc[valid]

    curves = []
    for column in prediction_columns:
        curves.append(decision_curve(y, work[column], threshold_values, model_name=column[:-5]))

    prevalence = float(y.mean()) if len(y) else np.nan
    n = len(y)
    for label, benefit in (("treat_all", None), ("treat_none", 0.0)):
        rows = []
        for threshold in threshold_values:
            if label == "treat_all":
                value = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
            else:
                value = benefit
            rows.append({"model": label, "threshold": threshold, "net_benefit": value})
        curves.append(pd.DataFrame(rows))

    result = pd.concat(curves, ignore_index=True)
    result["n"] = n
    result["prevalence"] = prevalence
    result["standardised_net_benefit"] = result["net_benefit"] / prevalence if prevalence > 0 else np.nan
    return result


def decision_curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Summarise the best net benefit and threshold range above treat-all/none."""
    required = {"model", "threshold", "net_benefit"}
    missing = required.difference(curves.columns)
    if missing:
        raise KeyError(f"Decision-curve table is missing columns: {sorted(missing)}")

    rows = []
    reference = curves[curves["model"].isin(["treat_all", "treat_none"])]
    for model, group in curves[~curves["model"].isin(["treat_all", "treat_none"])].groupby("model"):
        merged = group.merge(
            reference.pivot(index="threshold", columns="model", values="net_benefit"),
            left_on="threshold",
            right_index=True,
            how="left",
        )
        treat_all = merged.get("treat_all", pd.Series(np.nan, index=merged.index))
        treat_none = merged.get("treat_none", pd.Series(np.nan, index=merged.index))
        better = (merged["net_benefit"] > treat_all) & (merged["net_benefit"] > treat_none)
        best = group.loc[group["net_benefit"].idxmax()]
        rows.append(
            {
                "model": model,
                "best_threshold": best["threshold"],
                "maximum_net_benefit": best["net_benefit"],
                "threshold_min_better_than_references": merged.loc[better, "threshold"].min() if better.any() else np.nan,
                "threshold_max_better_than_references": merged.loc[better, "threshold"].max() if better.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_decision_curves(
    curves: pd.DataFrame,
    ax=None,
    models: Sequence[str] | None = None,
    include_references: bool = True,
):
    """Plot net benefit curves and return the matplotlib axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    selected = list(models) if models is not None else list(curves["model"].drop_duplicates())
    for model in selected:
        if not include_references and model in {"treat_all", "treat_none"}:
            continue
        group = curves[curves["model"] == model].sort_values("threshold")
        style = {"linestyle": "--", "color": "black"} if model == "treat_all" else {}
        if model == "treat_none":
            style = {"linestyle": ":", "color": "black"}
        ax.plot(group["threshold"], group["net_benefit"], label=model, **style)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend()
    ax.grid(alpha=0.2)
    return ax
