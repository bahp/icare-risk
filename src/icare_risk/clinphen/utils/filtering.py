import pandas as pd
import numpy as np

def match_codes(series: pd.Series,
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