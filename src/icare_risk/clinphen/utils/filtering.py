import pandas as pd

def match_codes(series: pd.Series,
                target_codes: list) -> pd.Series:
    """Returns a boolean mask for robust, case-insensitive string matching."""
    # No need to upper, done when creating the phenotypes
    # Inside the registry
    #targets = [str(c).upper() for c in target_codes]
    return series.astype(str).str.upper().isin(series)


def extract_numeric(df: pd.DataFrame,
                    code_col: str,
                    val_col: str,
                    target_codes: list) -> pd.Series:
    """Filters a dataframe by target codes and returns cleaned numeric values."""
    if df.empty or code_col not in df.columns or val_col not in df.columns:
        return pd.Series(dtype=float)

    mask = match_codes(df[code_col], target_codes)
    return pd.to_numeric(df.loc[mask, val_col], errors="coerce").dropna()