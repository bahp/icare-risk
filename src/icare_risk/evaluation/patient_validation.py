from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..models.prediction import predict_probability
from ..models.training import fit_model
from .metrics import compute_metrics
from ..data.feature_sets import resolve_feature_set
from ..models.training import encode_binary_target


def grouped_cross_validate_model(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    group_column: str,
    *,
    model_type: str = "logistic_regression",
    params: Mapping | None = None,
    target_mapping: Mapping | None = None,
    n_splits: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-validate while keeping all admissions from a patient in one fold."""
    work = df[list(features) + [target, group_column]].copy()
    work["__target_numeric__"] = encode_binary_target(work[target], positive_values=(target_mapping or {}).get("positive_values"), negative_values=(target_mapping or {}).get("negative_values"))
    work = work.dropna(subset=["__target_numeric__", group_column]).reset_index(drop=True)
    y = work["__target_numeric__"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(work[features], y, groups=work[group_column]), start=1):
        fitted = fit_model(work.iloc[train_idx], features, "__target_numeric__", model_type=model_type, params=params or {})
        probabilities = predict_probability(fitted, work.iloc[valid_idx], features)
        metrics = compute_metrics(y.iloc[valid_idx].to_numpy(), probabilities, threshold=threshold)
        rows.append({"fold": fold, "n_train": len(train_idx), "n_validation": len(valid_idx), "n_train_patients": work.iloc[train_idx][group_column].nunique(), "n_validation_patients": work.iloc[valid_idx][group_column].nunique(), **metrics})
    return pd.DataFrame(rows)
