from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological development, validation, and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict


def add_parsed_date(
    df: pd.DataFrame,
    date_column: str,
    output_column: str = "_admission_date",
) -> pd.DataFrame:
    """Return a copy with a parsed datetime column."""
    if date_column not in df.columns:
        raise KeyError(f"Date column not found: {date_column}")
    result = df.copy()
    result[output_column] = pd.to_datetime(result[date_column], errors="coerce")
    return result


def chronological_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    *,
    drop_missing_dates: bool = True,
    date_output_column: Optional[str] = None,
) -> TemporalSplit:
    """Split admissions chronologically using explicit calendar cut-offs.

    Intervals are: train < train_end, validation [train_end, validation_end),
    and test >= validation_end. Cut-offs must be selected without using test
    performance.
    """
    work = add_parsed_date(df, date_column, date_output_column or date_column)
    parsed_column = date_output_column or date_column
    if drop_missing_dates:
        work = work.loc[work[parsed_column].notna()].copy()
    elif work[parsed_column].isna().any():
        raise ValueError("Missing or unparseable dates are present")

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end")

    train = work.loc[work[parsed_column] < train_end].copy()
    validation = work.loc[
        (work[parsed_column] >= train_end) & (work[parsed_column] < validation_end)
    ].copy()
    test = work.loc[work[parsed_column] >= validation_end].copy()

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        boundaries={
            "date_column": date_column,
            "train_end": train_end,
            "validation_end": validation_end,
            "n_dropped_missing_dates": int(len(df) - len(work)),
        },
    )


def add_split_label(
    df: pd.DataFrame,
    date_column: str,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    output_column: str = "split",
) -> pd.DataFrame:
    """Add a chronological train/validation/test label.

    The intervals are ``train < train_end``, ``train_end <= validation <
    validation_end``, and ``test >= validation_end``. Missing or unparseable
    dates retain a missing split label and are not silently assigned to a
    period.

    Parameters
    ----------
    df:
        Admission-level input table.
    date_column:
        Column containing admission dates.
    train_end, validation_end:
        Calendar boundaries selected from the study design.
    output_column:
        Name of the generated split column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the split label added.
    """
    result = add_parsed_date(df, date_column)
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    result[output_column] = pd.NA
    result.loc[result["_admission_date"] < train_end, output_column] = "train"
    result.loc[
        (result["_admission_date"] >= train_end)
        & (result["_admission_date"] < validation_end),
        output_column,
    ] = "validation"
    result.loc[result["_admission_date"] >= validation_end, output_column] = "test"
    return result.drop(columns="_admission_date")


def split_summary(
    df: pd.DataFrame,
    split_column: str = "split",
    target_column: str | None = None,
    target_mapping: dict | None = None,
) -> pd.DataFrame:
    """Summarise admissions and event prevalence by temporal split.

    ``target_mapping`` should contain ``positive_values`` and
    ``negative_values`` when the outcome is stored as text. Unrecognised or
    missing outcomes are counted but excluded from event-rate calculations.
    """
    if split_column not in df.columns:
        raise KeyError(f"Split column not found: {split_column}")
    rows = []
    for split, group in df.groupby(split_column, dropna=False, sort=False):
        row = {"split": split, "n_admissions": len(group)}
        if target_column is not None and target_column in group.columns:
            if target_mapping:
                positive = {str(v).strip().lower() for v in target_mapping.get("positive_values", [])}
                negative = {str(v).strip().lower() for v in target_mapping.get("negative_values", [])}
                normalised = group[target_column].astype("string").str.strip().str.lower()
                values = pd.Series(pd.NA, index=group.index, dtype="Float64")
                values.loc[normalised.isin(positive)] = 1.0
                values.loc[normalised.isin(negative)] = 0.0
            else:
                values = pd.to_numeric(group[target_column], errors="coerce")
            row["n_non_missing_target"] = int(values.notna().sum())
            row["n_unknown_target"] = int(values.isna().sum())
            row["n_events"] = int(values.sum()) if values.notna().any() else 0
            row["event_rate"] = float(values.mean()) if values.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
