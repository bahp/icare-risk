from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _prefix_columns(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    return [column for column in columns if any(column.startswith(prefix) for prefix in prefixes)]


def resolve_feature_set(
    df: pd.DataFrame,
    definition: dict,
    *,
    exclude_columns: Iterable[str] = (),
) -> list[str]:
    """Resolve a declarative feature-set definition to existing dataframe columns.

    Supported definitions:
    - ``columns``: explicit column names
    - ``prefixes``: all columns beginning with one of the prefixes
    - ``include_all: true``: every column except exclusions
    """
    if not isinstance(definition, dict):
        raise TypeError("Feature-set definition must be a mapping")

    if "columns" in definition:
        selected = list(definition["columns"])
    elif "prefixes" in definition:
        selected = _prefix_columns(df.columns, definition["prefixes"])
    elif definition.get("include_all", False):
        selected = list(df.columns)
    else:
        raise ValueError("Feature set must define columns, prefixes, or include_all")

    excluded = set(exclude_columns) | set(definition.get("exclude", []))
    selected = [column for column in selected if column not in excluded]
    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not found: {missing}")
    if not selected:
        raise ValueError("Feature set resolved to zero columns")
    return selected


def feature_set_summary(
    df: pd.DataFrame,
    feature_sets: dict,
    *,
    exclude_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Return a tidy summary of configured feature sets."""
    rows = []
    for name, definition in feature_sets.items():
        columns = resolve_feature_set(df, definition, exclude_columns=exclude_columns)
        rows.append({
            "feature_set": name,
            "n_features": len(columns),
            "features": columns,
        })
    return pd.DataFrame(rows)
