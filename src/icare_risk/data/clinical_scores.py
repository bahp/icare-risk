from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def validate_score_columns(df: pd.DataFrame, score_columns: Iterable[str]) -> pd.DataFrame:
    """Validate configured clinical-score columns without modifying ``df``.

    Parameters
    ----------
    df:
        Admission-level dataset to inspect.
    score_columns:
        Column names expected to contain numeric score values.

    Returns
    -------
    pandas.DataFrame
        One row per configured column with existence, missingness, cardinality,
        and numeric range information.
    """
    rows = []
    for column in score_columns:
        if column not in df.columns:
            rows.append({
                "score": column,
                "exists": False,
                "n_missing": pd.NA,
                "n_unique": pd.NA,
                "min": pd.NA,
                "max": pd.NA
            })
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({
            "score": column,
            "exists": True,
            "n_missing": int(values.isna().sum()),
            "n_unique": int(values.nunique(dropna=True)),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)


def add_configured_scores(
    df: pd.DataFrame,
    score_definitions: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Add configured score columns to a copy of an admission-level table.

    A definition may use ``source_column`` to copy an existing score or
    ``constant`` for an explicit testing baseline. This function does not
    implement or infer score formulas.
    """
    result = df.copy()
    for name, definition in score_definitions.items():
        if "source_column" in definition:
            source = str(definition["source_column"])
            if source not in result.columns:
                raise KeyError(f"Score source column not found: {source}")
            result[name] = pd.to_numeric(result[source], errors="coerce")
        elif "constant" in definition:
            result[name] = definition["constant"]
        else:
            raise ValueError(
                f"Score '{name}' must define 'source_column' or 'constant'."
            )
    return result


def score_summary(df: pd.DataFrame, score_columns: Iterable[str]) -> pd.DataFrame:
    """Summarise score distributions for the analysis cohort."""
    rows = []
    for column in score_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({
            "score": column,
            "n": int(values.notna().sum()),
            "n_missing": int(values.isna().sum()),
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)
