from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def encode_binary_target(
    values: pd.Series,
    *,
    positive_values: Sequence | None = None,
    negative_values: Sequence | None = None,
) -> pd.Series:
    """Encode ESBL outcome labels as 1/0 and mark unresolved labels as missing.

    Labels such as ``unknown`` are deliberately returned as missing so they are
    excluded from supervised model fitting rather than treated as a negative.
    """
    positive_values = positive_values or [1, "1", True, "positive", "esbl", "esbl_positive"]
    negative_values = negative_values or [0, "0", False, "negative", "non-esbl", "non esbl", "non_esbl", "no esbl"]

    positive = {str(value).strip().lower() for value in positive_values}
    negative = {str(value).strip().lower() for value in negative_values}
    normalised = values.astype("string").str.strip().str.lower()

    encoded = pd.Series(pd.NA, index=values.index, dtype="Float64")
    encoded.loc[normalised.isin(positive)] = 1.0
    encoded.loc[normalised.isin(negative)] = 0.0
    return encoded


def build_estimator(model_type: str, params: Mapping | None = None):
    """Create a supported classifier without fitting it."""
    params = dict(params or {})
    if model_type in {"logistic_regression", "lr"}:
        defaults = {"max_iter": 2000, "class_weight": "balanced", "solver": "liblinear"}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model_type in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost to train XGBoost models") from exc
        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    raise ValueError(f"Unsupported model type: {model_type}")


def fit_model(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    model_type: str = "logistic_regression",
    params: Mapping | None = None,
    scale_numeric: bool = False,
    target_mapping: Mapping | None = None,
) -> Pipeline:
    """Fit a preprocessing and classification pipeline.

    Rows whose target cannot be mapped to binary 0/1 are excluded from fitting.
    This includes missing values and labels such as ``unknown`` unless explicitly
    included in ``target_mapping``.

    The preprocessing pipeline is fitted only on ``df`` supplied to this function.
    It performs imputation, categorical encoding, and optional numeric scaling
    before fitting the classifier.

    Parameters
    ----------
    df:
        Development data used for fitting.
    features:
        Predictor columns.
    target:
        Outcome column.
    model_type:
        ``"logistic_regression"`` or ``"xgboost"``.
    params:
        Classifier parameters.
    scale_numeric:
        Whether numeric predictors should be standardised.
    target_mapping:
        Mapping containing ``positive_values`` and ``negative_values``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted preprocessing and classifier pipeline.
"""
    features = list(features)
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    work = df[features + [target]].copy()
    target_mapping = dict(target_mapping or {})

    # Preserve an already encoded binary target. This is important for cross-
    # validation, where each fold uses a temporary 0/1 target column.
    numeric_target = pd.to_numeric(work[target], errors="coerce")
    is_binary_numeric = (
            numeric_target.notna().all()
            and set(numeric_target.astype(int).unique()).issubset({0, 1})
    )
    if is_binary_numeric:
        work["__binary_target__"] = numeric_target.astype(float)
    else:
        work["__binary_target__"] = encode_binary_target(
            work[target],
            positive_values=target_mapping.get("positive_values"),
            negative_values=target_mapping.get("negative_values"),
        )

    unresolved = work["__binary_target__"].isna().sum()
    work = work.dropna(subset=["__binary_target__"])
    if work["__binary_target__"].nunique() < 2:
        raise ValueError("Training target must contain both binary classes after target encoding")
    preprocessor = build_preprocessor(work, features, scale_numeric=scale_numeric)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", build_estimator(model_type, params)),
    ])
    if unresolved:
        import warnings
        warnings.warn(
            f"Excluded {unresolved} rows with unrecognised or unknown target labels from training.",
            UserWarning,
            stacklevel=2,
        )
    pipeline.fit(work[features], work["__binary_target__"].astype(int))
    return pipeline
