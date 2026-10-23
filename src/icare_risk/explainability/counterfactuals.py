from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def evaluate_counterfactuals(
    fitted_pipeline,
    X: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Evaluate named counterfactual feature modifications.

    Each scenario maps feature names to replacement values. Rows are copied and
    modified only for the named features. Results describe model behaviour under
    those hypothetical inputs and should not be interpreted causally.
    """
    baseline = fitted_pipeline.predict_proba(X)[:, 1]
    rows = [{
        "scenario": "baseline",
        "n_modified_features": 0,
        "mean_probability": baseline.mean(),
        "mean_change": 0.0,
        "median_change": 0.0,
    }]
    for name, changes in scenarios.items():
        missing = [column for column in changes if column not in X.columns]
        if missing:
            raise KeyError(f"Counterfactual '{name}' uses missing columns: {missing}")
        modified = X.copy()
        for column, value in changes.items():
            modified[column] = value
        prediction = fitted_pipeline.predict_proba(modified)[:, 1]
        change = prediction - baseline
        rows.append({
            "scenario": name,
            "n_modified_features": len(changes),
            "mean_probability": prediction.mean(),
            "mean_change": change.mean(),
            "median_change": float(pd.Series(change).median()),
        })
    return pd.DataFrame(rows)
