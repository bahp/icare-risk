import pandas as pd
import warnings

from typing import Iterable
from typing import Dict, Optional, List

# ------------------------------------------------------------------------
# Helper methods
# ------------------------------------------------------------------------
def validate_required_columns(
        df: pd.DataFrame,
        required_cols: List[str],
        score_name: str = "Score",
        verbose: bool = False,
) -> List[str]:
    """
    Generic helper to check for missing columns in a dataframe for any
    score function. Always flags missing columns either via warnings or
    verbose logging.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    required_cols : List[str]
        List of column names required for the score calculation.
    score_name : str, optional
        Name of the score for logging/warning context.
    verbose : bool, optional
        Whether verbose logging is enabled.
    logger : logging.Logger, optional
        Logger instance for audit traces.

    Returns
    -------
    List[str]
        A list of missing column names.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        warning_msg = f"[{score_name}] Missing expected columns in DataFrame: {missing_cols}"
        if verbose:
            print(f"    ⚠️ {warning_msg}")
        else:
            warnings.warn(warning_msg, UserWarning)

    return missing_cols

def validate_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """
    Validates that all required columns exist in the DataFrame.
    Raises a KeyError with a clear message if any are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available columns are: {df.columns.tolist()}"
        )