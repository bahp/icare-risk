from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from ..models.prediction import predict_probability
from ..models.training import encode_binary_target, fit_model
from .metrics import compute_metrics


def evaluate_model_cross_validation(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    model_type: str = "logistic_regression",
    params: Mapping | None = None,
    target_mapping: Mapping | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    threshold: float = 0.5,
    random_state: int = 42,
    scale_numeric: bool = False,
) -> pd.DataFrame:
    """Evaluate one model using repeated stratified cross-validation.

    Cross-validation is intended for development data only. Preprocessing and
    model fitting are repeated inside every fold to avoid information leakage.
    Rows with unresolved target labels are excluded before splitting.
    """
    features = list(features)
    missing = [column for column in [*features, target] if column not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    work = df[features + [target]].copy()
    work["__target_numeric__"] = encode_binary_target(
        work[target],
        positive_values=(target_mapping or {}).get("positive_values"),
        negative_values=(target_mapping or {}).get("negative_values"),
    )
    work = work.dropna(subset=["__target_numeric__"]).reset_index(drop=True)
    y = work["__target_numeric__"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Cross-validation target must contain both binary classes")
    if y.value_counts().min() < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} observations; counts are "
            f"{y.value_counts().to_dict()}"
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rows = []

    for fold_id, (train_idx, valid_idx) in enumerate(
        splitter.split(work[features], y), start=1
    ):
        fold_model = fit_model(
            work.iloc[train_idx],
            features,
            "__target_numeric__",
            model_type=model_type,
            params=params,
            scale_numeric=scale_numeric,
        )
        probabilities = predict_probability(
            fold_model,
            work.iloc[valid_idx],
            features,
        )
        metrics = compute_metrics(
            y.iloc[valid_idx],
            probabilities,
            threshold=threshold,
        )
        repeat = (fold_id - 1) // n_splits + 1
        fold = (fold_id - 1) % n_splits + 1
        rows.append({
            "repeat": repeat,
            "fold": fold,
            "n_train": len(train_idx),
            "n_validation": len(valid_idx),
            "model_type": model_type,
            **metrics,
        })

    return pd.DataFrame(rows)


def summarise_cross_validation(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise fold-level results using median and interquartile ranges."""
    metric_columns = [
        column for column in ["auroc", "auprc", "brier_score", "sensitivity", "specificity", "ppv", "npv"]
        if column in results.columns
    ]
    if not metric_columns:
        raise ValueError("No recognised metric columns found")

    summary = []
    for metric in metric_columns:
        values = pd.to_numeric(results[metric], errors="coerce").dropna()
        summary.append({
            "metric": metric,
            "n_folds": int(values.size),
            "median": float(values.median()) if len(values) else np.nan,
            "q1": float(values.quantile(0.25)) if len(values) else np.nan,
            "q3": float(values.quantile(0.75)) if len(values) else np.nan,
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan,
            "mean": float(values.mean()) if len(values) else np.nan,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        })
    return pd.DataFrame(summary)


def cross_validate_models(
    df: pd.DataFrame,
    model_configs: Sequence[Mapping],
    feature_sets: Mapping[str, Mapping],
    *,
    target: str,
    target_mapping: Mapping | None = None,
    exclude_columns: Sequence[str] = (),
    n_splits: int = 5,
    n_repeats: int = 3,
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated cross-validation for all configured models."""
    from ..data.feature_sets import resolve_feature_set

    fold_results = []
    summary_results = []
    for config in model_configs:
        name = config["name"]
        features = resolve_feature_set(
            df,
            feature_sets[config["feature_set"]],
            exclude_columns=exclude_columns,
        )
        folds = evaluate_model_cross_validation(
            df,
            features,
            target,
            model_type=config["type"],
            params=config.get("params", {}),
            target_mapping=target_mapping,
            n_splits=n_splits,
            n_repeats=n_repeats,
            threshold=threshold,
            random_state=random_state,
            scale_numeric=config.get("scale_numeric", False),
        )
        folds.insert(0, "model", name)
        folds.insert(1, "n_features", len(features))
        fold_results.append(folds)

        summary = summarise_cross_validation(folds)
        summary.insert(0, "model", name)
        summary.insert(1, "n_features", len(features))
        summary_results.append(summary)

    return pd.concat(fold_results, ignore_index=True), pd.concat(summary_results, ignore_index=True)
