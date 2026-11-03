from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise an informative error when required columns are absent."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_admission_table(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    admission_id: str = "SPELL_IDENTIFIER",
    date_columns: Iterable[str] = ("ADMISSION_DATE", "DISCHARGE_DATE"),
) -> dict:
    """Run structural checks on an admission-level table.

    Parameters
    ----------
    df:
        Admission-level input table.
    required_columns:
        Columns that must be present.
    admission_id:
        Identifier used to detect duplicate admission rows.
    date_columns:
        Date columns checked for unparseable non-missing values.

    Returns
    -------
    dict
        Counts, duplicate information, date parsing failures, and warnings.
        This function reports problems but does not modify the input table.
    """
    require_columns(df, required_columns)
    result = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "is_empty": df.empty,
        "admission_id_present": admission_id in df,
        "duplicate_admission_rows": None,
        "date_parse_failures": {},
        "warnings": [],
    }

    if admission_id in df:
        result["duplicate_admission_rows"] = int(df[admission_id].duplicated(keep=False).sum())
        if result["duplicate_admission_rows"]:
            result["warnings"].append(f"Duplicate values found in {admission_id}")

    for column in date_columns:
        if column in df:
            parsed = pd.to_datetime(df[column], errors="coerce")
            failures = int(df[column].notna().sum() - parsed.notna().sum())
            result["date_parse_failures"][column] = failures
            if failures:
                result["warnings"].append(f"{failures} values failed date parsing in {column}")

    return result


def validate_binary_target(df: pd.DataFrame, target_column: str) -> dict:
    """Describe whether a target column has two observed non-missing labels.

    This structural check does not decide which label is positive or negative;
    use the configured target mapping for that purpose. Values such as
    ``unknown`` are therefore reported in ``unique_values`` rather than
    silently treated as negative.
    """
    require_columns(df, [target_column])
    values = df[target_column].dropna().unique().tolist()
    return {
        "target_column": target_column,
        "n_missing": int(df[target_column].isna().sum()),
        "unique_values": values,
        "n_unique_non_missing": len(values),
        "is_binary": len(values) == 2,
    }
