from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_admission_file(path: str | Path, **read_kwargs) -> pd.DataFrame:
    """Load an admission-level CSV, TXT, or Parquet file.

    Parameters
    ----------
    path:
        Input dataset path. The file type is selected from its extension.
    **read_kwargs:
        Extra arguments passed to ``pandas.read_csv`` or
        ``pandas.read_parquet``. Do not pass ``parquet=True``; Parquet
        handling is inferred automatically.

    Returns
    -------
    pandas.DataFrame
        The loaded admission-level table.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Admission dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **read_kwargs)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, **read_kwargs)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-row dataset-level summary."""
    return pd.DataFrame([{
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_subjects": df["SUBJECT"].nunique() if "SUBJECT" in df else pd.NA,
        "n_admissions": df["SPELL_IDENTIFIER"].nunique() if "SPELL_IDENTIFIER" in df else len(df),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }])


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise type, missingness, uniqueness and basic value information."""
    rows = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "n_rows": len(series),
            "n_missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "n_unique": int(series.nunique(dropna=True)),
            "example_values": ", ".join(map(str, non_null.head(5).tolist())),
        })
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True])


def duplicate_summary(
    df: pd.DataFrame,
    admission_id_columns: Iterable[str] = ("SPELL_IDENTIFIER", "ENCNTR_ID", "EPISODE_IDENTIFIER"),
) -> pd.DataFrame:
    """Check duplicate admission identifiers."""
    rows = []
    for column in admission_id_columns:
        if column not in df:
            continue
        duplicated = df[column].duplicated(keep=False)
        rows.append({
            "identifier": column,
            "n_non_missing": int(df[column].notna().sum()),
            "n_unique": int(df[column].nunique(dropna=True)),
            "n_rows_in_duplicate_groups": int(duplicated.sum()),
            "has_duplicates": bool(duplicated.any()),
        })
    return pd.DataFrame(rows)


def outcome_summary(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Summarise binary or categorical outcome prevalence."""
    if target_column not in df:
        raise KeyError(f"Target column not found: {target_column}")
    counts = df[target_column].value_counts(dropna=False).rename_axis("value").reset_index(name="n")
    counts["percentage"] = counts["n"] / len(df) * 100
    return counts


def _binary_event_indicator(series: pd.Series) -> pd.Series:
    """Convert common binary outcome encodings to nullable numeric 0/1 values."""
    normalised = series.astype("string").str.strip().str.lower()
    positive = {"1", "true", "t", "yes", "y", "positive", "pos", "event"}
    negative = {"0", "false", "f", "no", "n", "negative", "neg", "non-event"}

    result = pd.Series(pd.NA, index=series.index, dtype="Float64")
    result.loc[normalised.isin(positive)] = 1.0
    result.loc[normalised.isin(negative)] = 0.0

    # Preserve other numeric 0/1 encodings, including mixed numeric/string data.
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_binary = numeric.where(numeric.isin([0, 1]))
    result = result.fillna(numeric_binary)
    return result


def temporal_summary(df: pd.DataFrame,
                     date_column: str,
                     target_column: str | None = None,
                     freq: str = "YS") -> pd.DataFrame:
    """Summarise admissions and, optionally, event rates over time.

    Outcome values are coerced to numeric binary values so that CSV-loaded
    values such as ``"0"`` and ``"1"`` do not cause string arithmetic errors.
    Unrecognised or missing outcome values are excluded from ``n_events`` and
    reported in ``n_missing_outcome``.
    """
    if date_column not in df:
        raise KeyError(f"Date column not found: {date_column}")
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column])
    grouped = work.groupby(pd.Grouper(key=date_column, freq=freq))
    result = grouped.size().rename("n_admissions").to_frame()

    if target_column is not None and target_column in work:
        work["_event_indicator"] = _binary_event_indicator(work[target_column])
        indicator_grouped = work.groupby(pd.Grouper(key=date_column, freq=freq))["_event_indicator"]
        result["n_events"] = indicator_grouped.sum(min_count=1)
        result["n_missing_outcome"] = indicator_grouped.apply(lambda values: int(values.isna().sum()))
        result["event_rate"] = result["n_events"] / result["n_admissions"]

    return result.reset_index(names="period")


def save_audit_report(report: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Save audit tables as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
