"""Feature ranking and parsimonious-model utilities.

Feature selection should be performed using development data only. The temporal
hold-out test set must not be used to choose the selected features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _raw_feature_name(transformed_name: str, raw_features: Sequence[str]) -> str | None:
    """Map a transformed sklearn feature name back to its raw input feature."""
    name = str(transformed_name)
    candidates = sorted((str(feature) for feature in raw_features), key=len, reverse=True)
    for feature in candidates:
        if name == feature or name.endswith(f"__{feature}") or name.endswith(f"_{feature}"):
            return feature
    return None


def rank_pipeline_features(
    fitted_pipeline,
    raw_features: Sequence[str],
    *,
    method: str = "model",
) -> pd.DataFrame:
    """Rank raw input features using a fitted logistic or tree pipeline.

    One-hot encoded columns are aggregated back to their original raw feature.
    ``method='model'`` uses absolute logistic coefficients or native tree
    importance. This ranking is intended for development-data feature selection.
    """
    classifier = fitted_pipeline.named_steps["classifier"]
    transformed_names = list(
        fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    if hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_)[0])
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
    else:
        raise TypeError("Pipeline classifier must expose coef_ or feature_importances_.")

    rows = []
    for transformed, value in zip(transformed_names, values):
        raw = _raw_feature_name(transformed, raw_features)
        if raw is not None:
            rows.append({"feature": raw, "importance": float(value)})

    if not rows:
        raise ValueError("Could not map transformed features to raw input features.")

    table = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "feature", "importance"]]


def select_top_features(
    ranking: pd.DataFrame,
    n_features: int,
    *,
    feature_column: str = "feature",
) -> list[str]:
    """Return the top ``n_features`` from a ranking table."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if feature_column not in ranking.columns:
        raise KeyError(f"Ranking table lacks '{feature_column}'.")
    return ranking.sort_values("rank")[feature_column].head(n_features).tolist()


def feature_set_table(
    ranking: pd.DataFrame,
    sizes: Sequence[int] = (5, 10, 20),
) -> dict[str, list[str]]:
    """Create named top-k feature sets from a ranking table."""
    available = len(ranking)
    result = {"full": ranking.sort_values("rank")["feature"].tolist()}
    for size in sizes:
        if size <= available:
            result[f"top_{size}"] = select_top_features(ranking, size)
    return result
