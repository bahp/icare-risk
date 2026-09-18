# Libraries
import pandas as pd
import numpy as np

def pitt_fever_score(df: pd.DataFrame,
                     temp_col: str = "temp_worst_deviation") -> pd.Series:
    """Vectorized Pitt Fever Score."""
    if temp_col not in df.columns:
        return pd.Series(0, index=df.index)

    temps = pd.to_numeric(df[temp_col], errors='coerce')
    conditions = [
        (temps <= 35.0) | (temps >= 40.0),
        (temps >= 35.1) | (temps <= 36.0),
        (temps >= 39.0) & (temps <= 39.9)
    ]
    choices = [2, 1, 1]

    # Assigns 2, 1, or defaults to 0 based on the conditions
    return pd.Series(np.select(conditions, choices, default=0), index=df.index)


def pitt_mental_score(df: pd.DataFrame,
                      gcs_min_col: str = "gcs_min") -> pd.Series:
    """Vectorized Pitt Mental Score based on Glasgow Coma Scale."""
    if gcs_min_col not in df.columns:
        return pd.Series(0, index=df.index)

    gcs = pd.to_numeric(df[gcs_min_col], errors='coerce')
    conditions = [
        (gcs <= 9),
        (gcs >= 10) & (gcs <= 12),
        (gcs >= 13) & (gcs <= 14)
    ]
    choices = [4, 2, 1]

    # Assigns 4, 2, 1, or defaults to 0 based on GCS
    return pd.Series(np.select(conditions, choices, default=0), index=df.index)