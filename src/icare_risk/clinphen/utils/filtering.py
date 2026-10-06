import re
import pandas as pd
from typing import List, Set, Union

def _match_exact(s: pd.Series, targets: List[str]) -> pd.Series:
    """Fast-path hash set lookup."""
    return s.isin(set(targets))


def _match_prefix(s: pd.Series, targets: List[str]) -> pd.Series:
    """Fast-path fixed slice if uniform length, fallback to anchored regex."""
    target_set = set(targets)
    lengths = {len(t) for t in target_set}

    # Fast path: Uniform length prefix slicing
    if len(lengths) == 1:
        k = next(iter(lengths))
        return s.str[:k].isin(target_set)

    # Fallback: Single-pass anchored regex
    pattern = f"^(?:{'|'.join(re.escape(t) for t in targets)})"
    return s.str.match(pattern, na=False)


def _match_contains(s: pd.Series, targets: List[str]) -> pd.Series:
    """Sub-string matching."""
    pattern = f"(?:{'|'.join(re.escape(t) for t in targets)})"
    return s.str.contains(pattern, na=False, regex=True)


def _match_regex(s: pd.Series,
                 targets: List[str],
                 case_sensitive: bool) -> pd.Series:
    """Raw regular expression matching."""
    pattern = "|".join(f"(?:{t})" for t in targets)
    flags = 0 if case_sensitive else re.IGNORECASE
    return s.str.contains(pattern, na=False, regex=True, flags=flags)

def match_codes(
    series: pd.Series,
    target_codes: Union[List[str], Set[str], str],
    match_type: str = "exact",
    case_sensitive: bool = False,
    strip_dots: bool = True
) -> pd.Series:
    """Dispatches matching strategy based on configuration.

    .. note: Would numpy be much faster?
    """
    if series.empty or not target_codes:
        return pd.Series(False, index=series.index)

    targets = [str(c)
        for c in ([target_codes] if isinstance(target_codes, str) else target_codes) if c]
    if not targets:
        return pd.Series(False, index=series.index)

    s = series.fillna('').astype(str)
    if not case_sensitive:
        s = s.str.upper()
        targets = [t.upper() for t in targets]

    if strip_dots and match_type in ("exact", "prefix", "startswith"):
        s = s.str.replace(r'[\.\s]+', '', regex=True)
        targets = [re.sub(r'[\.\s]+', '', t) for t in targets]

    # Modular Dispatchers
    if match_type == "exact":
        return _match_exact(s, targets)
    elif match_type in ("prefix", "startswith"):
        return _match_prefix(s, targets)
    elif match_type == "contains":
        return _match_contains(s, targets)
    elif match_type == "regex":
        return _match_regex(s, targets, case_sensitive)
    else:
        raise ValueError(f"Unknown match_type: {match_type}")

def match_codes_old(series: pd.Series,
                target_codes: list) -> pd.Series:
    """Returns a boolean mask for robust, case-insensitive string matching.

    Will numpy be fasteR?
    """
    #targets = [str(c).upper() for c in target_codes] # Needed
    #return series.astype(str).str.upper().isin(targets)
    return series.isin(target_codes)


def extract_numeric(df: pd.DataFrame,
                    code_col: str,
                    val_col: str,
                    target_codes: list) -> pd.Series:
    """Filters a dataframe by target codes and returns cleaned numeric values."""
    if df.empty or code_col not in df.columns or val_col not in df.columns:
        return pd.Series(dtype=float)

    mask = match_codes(df[code_col], target_codes)
    return pd.to_numeric(df.loc[mask, val_col], errors="coerce").dropna()