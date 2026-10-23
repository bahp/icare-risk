from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def _binary_target(values: pd.Series, positive_values=None, negative_values=None) -> pd.Series:
    positive = {str(v).strip().lower() for v in (positive_values or [1, "1", "positive", "esbl"])}
    negative = {str(v).strip().lower() for v in (negative_values or [0, "0", "negative", "non-esbl", "non esbl"])}
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    numeric_mask = numeric.isin([0, 1])
    result.loc[numeric_mask] = numeric.loc[numeric_mask]
    labels = values.astype("string").str.strip().str.lower()
    result.loc[labels.isin(positive)] = 1.0
    result.loc[labels.isin(negative)] = 0.0
    return result


def evaluate_raw_scores(
    df: pd.DataFrame,
    score_columns: Sequence[str] | Mapping[str, str],
    target_column: str,
    *,
    target_mapping: Mapping | None = None,
    subset: pd.Series | None = None,
    thresholds: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Evaluate existing clinical scores directly, without refitting them."""
    if isinstance(score_columns, Mapping):
        scores = dict(score_columns)
    else:
        scores = {column: column for column in score_columns}
    work = df.loc[subset].copy() if subset is not None else df.copy()
    y = _binary_target(
        work[target_column],
        (target_mapping or {}).get("positive_values"),
        (target_mapping or {}).get("negative_values"),
    )
    rows = []
    for name, column in scores.items():
        if column not in work.columns:
            raise KeyError(f"Score column not found: {column}")
        valid = y.notna() & pd.to_numeric(work[column], errors="coerce").notna()
        yt = y.loc[valid].astype(int)
        score = pd.to_numeric(work.loc[valid, column], errors="coerce")
        row = {"score": name, "source_column": column, "n": len(score), "n_events": int(yt.sum())}
        row["auroc"] = roc_auc_score(yt, score) if yt.nunique() == 2 else np.nan
        row["auprc"] = average_precision_score(yt, score) if yt.nunique() == 2 else np.nan
        for threshold in (thresholds or {}).get(name, []):
            predicted = score >= threshold
            tp = int((predicted & yt.eq(1)).sum())
            tn = int((~predicted & yt.eq(0)).sum())
            fp = int((predicted & yt.eq(0)).sum())
            fn = int((~predicted & yt.eq(1)).sum())
            row[f"threshold_{threshold}_sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
            row[f"threshold_{threshold}_specificity"] = tn / (tn + fp) if tn + fp else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_score_probability_calibrator(scores: pd.Series, y: pd.Series) -> LogisticRegression:
    """Fit logistic calibration for a score using development data only."""
    x = pd.to_numeric(scores, errors="coerce").to_numpy().reshape(-1, 1)
    target = pd.to_numeric(y, errors="coerce")
    valid = np.isfinite(x[:, 0]) & target.notna()
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(x[valid], target.loc[valid].astype(int))
    return calibrator
