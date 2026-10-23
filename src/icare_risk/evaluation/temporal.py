from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_metrics, encode_binary_target

def _numeric_target(
    values: pd.Series,
    target_mapping: dict | None = None,
) -> pd.Series:
    """Convert common binary outcome encodings to numeric 0/1 values."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        mapped = numeric.copy()
    else:
        mapped = pd.Series(pd.NA, index=values.index, dtype="Float64")

    mapping = target_mapping or {}
    positive = {str(value).strip().lower() for value in mapping.get("positive_values", [])}
    negative = {str(value).strip().lower() for value in mapping.get("negative_values", [])}
    text = values.astype("string").str.strip().str.lower()
    mapped = mapped.mask(text.isin(positive), 1)
    mapped = mapped.mask(text.isin(negative), 0)
    return pd.to_numeric(mapped, errors="coerce")

def _prediction_columns(df: pd.DataFrame, prediction_columns: Sequence[str] | None) -> list[str]:
    columns = list(prediction_columns or [c for c in df.columns if c.endswith("_prob")])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Prediction columns not found: {missing}")
    return columns


def evaluate_over_time(
    df: pd.DataFrame,
    *,
    date_column: str,
    target_column: str,
    prediction_columns: Sequence[str] | None = None,
    freq: str = "Q",
    threshold: float = 0.5,
    target_mapping: dict | None = None,
    min_n: int = 1,
) -> pd.DataFrame:
    """Evaluate every prediction column within chronological calendar bins.

    Rows with an unresolved target or invalid prediction are excluded by
    ``compute_metrics``. ``min_n`` prevents reporting very small periods.
    """
    required = {date_column, target_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Required columns not found: {sorted(missing)}")

    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column]).sort_values(date_column)
    columns = _prediction_columns(work, prediction_columns)
    rows: list[dict] = []

    for period, group in work.groupby(pd.Grouper(key=date_column, freq=freq), sort=True):
        if len(group) < min_n:
            continue
        for column in columns:
            metrics = compute_metrics(
                group[target_column], group[column],
                threshold=threshold,
                target_mapping=target_mapping,
            )
            rows.append({
                "period": period,
                "period_label": period.strftime("%Y-%m-%d"),
                "model": column.removesuffix("_prob"),
                "prediction_column": column,
                **metrics,
            })
    return pd.DataFrame(rows)


def temporal_split_summary(
    df: pd.DataFrame,
    *,
    date_column: str,
    target_column: str,
    split_column: str = "split",
    target_mapping: dict | None = None,
    freq: str = "Q",
) -> pd.DataFrame:
    """Summarise admissions and resolved events by split and calendar period."""
    required = {date_column, target_column, split_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Required columns not found: {sorted(missing)}")

    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work["_target_numeric"] = encode_binary_target(work[target_column], target_mapping)
    work = work.dropna(subset=[date_column])
    # ``YS``/``YE`` are valid Grouper frequencies but not valid Period
    # frequencies in all pandas versions, so create bins with Grouper.
    work["period"] = work[date_column].dt.to_period("Y").astype(str) if freq in {"Y", "YS", "YE", "A", "AS", "AY"} else work[date_column].dt.to_period(freq).astype(str)

    result = (
        work.groupby([split_column, "period"], dropna=False)
        .agg(
            n_rows=(target_column, "size"),
            n_resolved=("_target_numeric", "count"),
            n_events=("_target_numeric", "sum"),
        )
        .reset_index()
    )
    result["n_non_events"] = result["n_resolved"] - result["n_events"]
    result["event_rate"] = result["n_events"].div(result["n_resolved"].replace(0, np.nan))
    return result


def temporal_event_rate_summary(
    df: pd.DataFrame,
    *,
    date_column: str,
    target_column: str,
    freq: str = "YS",
    target_mapping: dict | None = None,
) -> pd.DataFrame:
    """Summarise outcome prevalence by calendar period, ignoring split labels.

    This deliberately groups only by calendar period. It prevents partial
    train/validation/test years from producing multiple points for the same
    year. Unknown outcomes are excluded from the event-rate denominator.

    Returns
    -------
    pandas.DataFrame
        One row per calendar period with ``n_rows``, ``n_resolved``,
        ``n_events``, ``n_non_events``, and ``event_rate``.
    """
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column])
    work["_target_numeric"] = _numeric_target(
        work[target_column], target_mapping
    )
    work["period"] = work[date_column].dt.to_period(
        freq.replace("S", "")
    ).dt.start_time

    result = (
        work.groupby("period", dropna=False)
        .agg(
            n_rows=("_target_numeric", "size"),
            n_resolved=("_target_numeric", "count"),
            n_events=("_target_numeric", lambda s: (s == 1).sum()),
            n_non_events=("_target_numeric", lambda s: (s == 0).sum()),
        )
        .reset_index()
    )
    result["event_rate"] = result["n_events"] / result["n_resolved"]
    return result


def save_temporal_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Save temporal results as CSV, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return output
