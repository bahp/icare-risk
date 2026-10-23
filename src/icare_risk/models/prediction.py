from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def predict_probability(model, df: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """Return positive-class probabilities with the original dataframe index."""
    features = list(features)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise KeyError(f"Prediction columns not found: {missing}")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The fitted model must implement predict_proba")
    probabilities = model.predict_proba(df[features])[:, 1]
    return pd.Series(probabilities, index=df.index, name="prediction")


def generate_prediction_table(
    df: pd.DataFrame,
    fitted_models: Mapping[str, object],
    feature_map: Mapping[str, Sequence[str]],
    *,
    id_columns: Sequence[str] = (),
    target_column: str | None = None,
    passthrough_columns: Sequence[str] = (),
    retain_all_columns: bool = True,
    sort_by_model: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Append model probabilities to the admission-level dataframe.

    By default, every original column is retained and each model probability is
    appended as ``<model>_prob``. Set ``retain_all_columns=False`` to create a
    compact table containing only identifiers, target, standard metadata, and
    explicitly requested ``passthrough_columns``.

    Parameters
    ----------
    retain_all_columns:
        Preserve all columns from ``df``. Recommended for subgroup and
        explainability analyses.
    sort_by_model:
        Model name whose ``<model>_prob`` column should determine row ordering.
        If ``None``, preserve the input order.
    ascending:
        Sort direction. Defaults to descending so highest predicted risk appears first.
    """
    if retain_all_columns:
        result = df.copy()
    else:
        columns = [column for column in id_columns if column in df.columns]
        if target_column and target_column in df.columns:
            columns.append(target_column)
        for column in ("ADMISSION_DATE", "split", *passthrough_columns):
            if column in df.columns and column not in columns:
                columns.append(column)
        result = df[columns].copy()

    for name, model in fitted_models.items():
        result[f"{name}_prob"] = predict_probability(model, df, feature_map[name])

    if sort_by_model is not None:
        if sort_by_model not in fitted_models:
            raise KeyError(
                f"Unknown model '{sort_by_model}'. Available models: "
                f"{list(fitted_models)}"
            )
        result = result.sort_values(
            f"{sort_by_model}_prob",
            ascending=ascending,
            kind="stable",
        ).reset_index(drop=True)

    return result
