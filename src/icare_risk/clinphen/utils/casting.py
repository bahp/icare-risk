"""Transparent sanity casting so phenotype authors never touch messy raw values."""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_NUMERIC_STRIP_RE = re.compile(r"[^\d.\-eE]")
_NULL_TOKENS = {"", "n/a", "na", "none", "null", "unknown", "nan", "-", "--"}


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Robustly coerce a messy lab/vitals result column to float64."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def _clean(v):
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in _NULL_TOKENS:
            return None
        s = s.replace(",", "")
        s = _NUMERIC_STRIP_RE.sub("", s)
        if s in ("", "-", ".", "-."):
            return None
        return s

    cleaned = series.map(_clean)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_datetime(series: pd.Series) -> pd.Series:
    """Permissive datetime parser: unparseable values become NaT, never raise."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def normalize_frame(
    df: pd.DataFrame,
    timestamp_cols: Iterable[str] = ("timestamp",),
    numeric_cols: Iterable[str] = ("value",),
) -> pd.DataFrame:
    """Apply standard datetime parsing + numeric coercion to logical columns."""
    df = df.copy()
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = parse_datetime(df[col])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    return df
