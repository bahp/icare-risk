import pandas as pd
import numpy as np


def sirs_tachycardia(df: pd.DataFrame,
                     hr_max_col: str = "heart_rate_max") -> pd.Series:
    """Vectorized SIRS Tachycardia: HR > 90[cite: 5]."""
    if hr_max_col not in df.columns:
        return pd.Series(0, index=df.index)
    return (df[hr_max_col] > 90.0).astype(int)


def sirs_tachypnea(df: pd.DataFrame,
                   rr_max_col: str = "resp_rate_max",
                   pco2_min_col: str = "pco2_min") -> pd.Series:
    """Vectorized SIRS Tachypnea: RR > 20 or pCO2 < 32[cite: 5]."""
    flag = pd.Series(False, index=df.index)

    if rr_max_col in df.columns:
        flag = flag | (df[rr_max_col] > 20.0)
    if pco2_min_col in df.columns:
        flag = flag | (df[pco2_min_col] < 32.0)

    return flag.astype(int)


def sirs_abnormal_temp(df: pd.DataFrame,
                       temp_max_col: str = "temp_max",
                       temp_min_col: str = "temp_min") -> pd.Series:
    """Vectorized SIRS Temp: > 38.0 or < 36.0[cite: 5]."""
    flag = pd.Series(False, index=df.index)

    if temp_max_col in df.columns:
        flag = flag | (df[temp_max_col] > 38.0)
    if temp_min_col in df.columns:
        flag = flag | (df[temp_min_col] < 36.0)

    return flag.astype(int)


def sirs_abnormal_wbc(
        df: pd.DataFrame,
        wbc_max_col: str = "wbc_max",
        wbc_min_col: str = "wbc_min",
        bands_max_col: str = "bands_max"
) -> pd.Series:
    """Vectorized SIRS WBC: > 12.0, < 4.0, or Bands > 10.0[cite: 5]."""
    flag = pd.Series(False, index=df.index)

    if wbc_max_col in df.columns:
        flag = flag | (df[wbc_max_col] > 12.0)
    if wbc_min_col in df.columns:
        flag = flag | (df[wbc_min_col] < 4.0)
    if bands_max_col in df.columns:
        flag = flag | (df[bands_max_col] > 10.0)

    return flag.astype(int)