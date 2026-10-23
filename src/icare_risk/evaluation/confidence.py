from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd
from .metrics import _valid_binary_data, compute_metrics

def bootstrap_metrics(
    y_true, y_prob, *, n_bootstraps=2000, ci=0.95, threshold=0.5,
    target_mapping: Mapping | None = None, random_state=42,
) -> dict:
    """Estimate percentile bootstrap intervals on a fixed evaluation set."""
    y, p = _valid_binary_data(y_true, y_prob, target_mapping)
    point = compute_metrics(y, p, threshold=threshold)
    interval_keys = [
        k for k in point if k not in {
            "n", "n_events", "event_rate", "threshold",
            "tp", "tn", "fp", "fn",
        }
    ]
    samples = {key: [] for key in interval_keys}
    if len(y) == 0:
        return {
            **point,
            **{
                f"{key}_ci_{bound}": np.nan
                for key in interval_keys
                for bound in ("lower", "upper")
            },
        }

    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        result = compute_metrics(y[indices], p[indices], threshold=threshold)
        for key in interval_keys:
            if pd.notna(result.get(key)):
                samples[key].append(result[key])

    alpha = (1 - ci) / 2
    output = dict(point)
    for key, values in samples.items():
        output[f"{key}_ci_lower"] = np.quantile(values, alpha) if values else np.nan
        output[f"{key}_ci_upper"] = np.quantile(values, 1 - alpha) if values else np.nan
    output["n_bootstraps_requested"] = n_bootstraps
    output["n_bootstraps_valid"] = max((len(v) for v in samples.values()), default=0)
    return output

def bootstrap_prediction_table(
    df: pd.DataFrame, *, target_column: str, prediction_columns=None,
    subset=None, n_bootstraps=2000, ci=0.95, threshold=0.5,
    target_mapping=None, random_state=42,
) -> pd.DataFrame:
    """Return point estimates and bootstrap intervals for each model column."""
    work = df.loc[subset] if subset is not None else df
    columns = list(prediction_columns or [c for c in work.columns if c.endswith("_prob")])
    rows = []
    for offset, column in enumerate(columns):
        result = bootstrap_metrics(
            work[target_column], work[column],
            n_bootstraps=n_bootstraps, ci=ci, threshold=threshold,
            target_mapping=target_mapping, random_state=random_state + offset,
        )
        rows.append({
            "model": column.removesuffix("_prob"),
            "prediction_column": column,
            **result,
        })
    return pd.DataFrame(rows)
