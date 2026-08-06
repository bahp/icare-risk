import pandas as pd
import numpy as np

def calculate_local_ward_triage_score(df,
                                      age_col='AGE_AT_ADMISSION',
                                      stress_col='local_stress_flag',
                                      **kwargs):
    """
    A custom local risk score for testing.
    Calculates a simple 0-5 point score based on age and the local stress phenotype.
    """
    score = pd.Series(0, index=df.index)

    # Rule 1: +2 points if patient is over 65
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors='coerce')
        score += np.where(age > 65, 2, 0)

    # Rule 2: +3 points if they meet our custom stress phenotype
    if stress_col in df.columns:
        stress_flag = pd.to_numeric(df[stress_col], errors='coerce')
        score += np.where(stress_flag == 1, 3, 0)

    return score.values