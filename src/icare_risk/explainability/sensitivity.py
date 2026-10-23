from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def prediction_sensitivity(
    fitted_pipeline,
    X: pd.DataFrame,
    feature: str,
    values: Sequence,
) -> pd.DataFrame:
    """Measure prediction changes when one feature is set to specified values.

    This is a model sensitivity analysis, not a causal effect estimate. Categorical
    features can be tested with replacement values; numeric features can be tested
    at clinically relevant reference values.
    """
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not found in X")
    baseline = fitted_pipeline.predict_proba(X)[:, 1]
    rows = [{"scenario": "baseline", "value": None, "mean_probability": baseline.mean(), "mean_change": 0.0}]
    for value in values:
        modified = X.copy()
        modified[feature] = value
        prediction = fitted_pipeline.predict_proba(modified)[:, 1]
        rows.append({
            "scenario": f"{feature}={value}",
            "value": value,
            "mean_probability": prediction.mean(),
            "mean_change": (prediction - baseline).mean(),
            "median_change": float(pd.Series(prediction - baseline).median()),
        })
    return pd.DataFrame(rows)


def one_at_a_time_sensitivity(
    fitted_pipeline,
    X: pd.DataFrame,
    scenarios: Mapping[str, Sequence],
) -> pd.DataFrame:
    """Run prediction_sensitivity for multiple features and concatenate results."""
    tables = []
    for feature, values in scenarios.items():
        table = prediction_sensitivity(fitted_pipeline, X, feature, values)
        table.insert(0, "feature", feature)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
