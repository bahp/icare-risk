from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from IPython.display import display


def inspect_df(
    df: pd.DataFrame,
    name: str,
    *,
    keys: Iterable[str] | None = None,
    columns: Iterable[str] | None = None,
    n: int = 5,
    show_dtypes: bool = False,
) -> None:
    """Display a compact, notebook-friendly dataframe diagnostic."""
    print(f"\n{'=' * 72}")
    print(f"CHECK: {name}")
    print(f"{'=' * 72}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    if keys is not None:
        keys = list(keys)
        missing_keys = [column for column in keys if column not in df.columns]

        if missing_keys:
            print(f"Missing key columns: {missing_keys}")
        else:
            duplicate_rows = int(df.duplicated(keys).sum())
            unique_keys = int(df[keys].drop_duplicates().shape[0])
            print(f"Unique keys: {unique_keys:,}")
            print(f"Duplicate rows by {keys}: {duplicate_rows:,}")

    if columns is None:
        preview = df.head(n)
    else:
        selected_columns = [
            column for column in columns if column in df.columns
        ]
        missing_columns = [
            column for column in columns if column not in df.columns
        ]

        if missing_columns:
            print(f"Missing preview columns: {missing_columns}")

        preview = df[selected_columns].head(n)

    print("\nPreview:")
    display(preview)

    missingness = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    missingness = missingness[missingness.gt(0)].head(10)

    if not missingness.empty:
        print("\nMissingness (%; top 10):")
        display(
            missingness
            .rename("missing_%")
            .round(1)
            .to_frame()
        )

    if show_dtypes:
        print("\nDtypes:")
        display(df.dtypes.rename("dtype").to_frame())


def check_value_counts(
    df: pd.DataFrame,
    column: str,
    *,
    name: str | None = None,
    n: int = 10,
) -> None:
    """Display the most common values in one column."""
    title = name or column
    print(f"\nValue check: {title}")

    if column not in df.columns:
        print(f"Column not found: {column}")
        return

    counts = df[column].value_counts(dropna=False).head(n)
    display(counts.rename("count").to_frame())


def check_merge_keys(
    df: pd.DataFrame,
    name: str,
    keys: Sequence[str],
) -> None:
    """Check that a dataframe has one row per merge key."""
    keys = list(keys)
    missing_keys = [column for column in keys if column not in df.columns]

    if missing_keys:
        raise KeyError(f"{name} is missing key columns: {missing_keys}")

    duplicate_rows = int(df.duplicated(keys).sum())
    unique_keys = int(df[keys].drop_duplicates().shape[0])

    print(
        f"{name}: {len(df):,} rows | "
        f"{unique_keys:,} unique keys | "
        f"{duplicate_rows:,} duplicate rows"
    )

    if duplicate_rows:
        raise ValueError(
            f"{name} is not unique on {keys}. "
            f"Found {duplicate_rows:,} duplicate rows."
        )


def check_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    *,
    name: str = "merge",
) -> None:
    """Check both sides of a merge before combining feature tables."""
    keys = list(keys)

    print(f"\nMerge check: {name}")
    check_merge_keys(left, "left", keys)
    check_merge_keys(right, "right", keys)


# Backwards-compatible aliases.
check_df = inspect_df
check_merge_keys = check_merge
