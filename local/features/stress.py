import pandas as pd
import numpy as np


def derive_local_stress_phenotype(df, hr_col='HR', temp_col='Temp', **kwargs):
    """
    A locally defined phenotype for testing the pipeline.
    Flags patients (1/0) who have both a high heart rate (>100) and fever (>38.0).
    """
    # Start by assuming the condition is absent (0)
    flag = pd.Series(0, index=df.index)

    # Safely evaluate if the required columns exist
    if hr_col in df.columns and temp_col in df.columns:
        hr = pd.to_numeric(df[hr_col], errors='coerce')
        temp = pd.to_numeric(df[temp_col], errors='coerce')

        # Apply the clinical logic
        flag.loc[(hr > 100) & (temp > 38.0)] = 1

    return flag.values