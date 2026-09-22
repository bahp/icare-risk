import pandas as pd
import numpy as np

from typing import Dict, Optional, List

def calculate_increment_esbl(df,
                             age_col='AGE_AT_ADMISSION',
                             charlson_col='charlson_quan_score',
                             pitt_col='pitt_score',
                             sirs_col='sirs_count',
                             bsi_not_urinary_col='increment_bsi_not_urinary_flag',
                             is_non_ecoli_col='increment_is_non_ecoli_flag',
                             inapprop_abx_col='increment_abx_inappropriate_flag',
                             **kwargs):
    """Computes the INCREMENT-ESBL predictive score for mortality.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    !!! warning "Mixed Input Expectations"
        This final predictive score requires a mix of **raw data** (e.g., patient age) and
        **intermediate computed scores** (e.g., total Charlson and Pitt scores). Ensure
        upstream intermediate scores are fully calculated before running this function.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        This score is designed to predict 30-day mortality in patients with bloodstream
        infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
        Enterobacteriaceae. The function calculates a total score based on demographic,
        clinical severity, microbiological, and treatment variables.

        | Clinical Variable                      | Condition Evaluated               | Points |
        | :------------------------------------- | :-------------------------------- | :----: |
        | Demographics                           | Age > 50 years                    |   +3   |
        | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
        | Acute Underlying Severity              | High Pitt bacteremia score (>= 6) |   +3   |
        | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
        | SIRS Severity                          | Severe SIRS / Shock (SIRS >= 2)   |   +4   |
        | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
        | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

        **References:** Palacios-Baena Z, et al. Development and validation of the INCREMENT-ESBL
        predictive score for mortality in patients with bloodstream infections. J Antimicrob
        Chemother. 2017;72(3):906-913.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='AGE_AT_ADMISSION'
        Column name for patient age (numeric).
    charlson_col : str, default='charlson_score'
        Column name for Charlson Comorbidity Index (numeric).
    pitt_col : str, default='pitt_score'
        Column name for Pitt bacteremia score (numeric).
    sirs_col : str, default='sirs_count'
        Column name for SIRS criteria count (numeric).
    bsi_not_urinary_col : str, default='increment_bsi_not_urinary_flag'
        Column name for non-urinary infection source flag (binary 1/0).
    is_non_ecoli_col : str, default='increment_non_ecoli_flag'
        Column name for non-E. coli isolated organism flag (binary 1/0).
    inapprop_abx_col : str, default='increment_inapprop_abx_flag'
        Column name for inappropriate antibiotic therapy flag (binary 1/0).

    Returns
    -------
    pd.Series
        A pandas Series containing the computed INCREMENT-ESBL score (integers)
        for each patient, matching the input DataFrame index.
    """
    pass