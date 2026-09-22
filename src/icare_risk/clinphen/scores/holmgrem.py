import pandas as pd
import numpy as np

from typing import Dict, Optional, List

def calculate_holmgren_score(df,
                             hosp_abroad_col='hx_hosp_abroad_12m',
                             prev_culture_col='hx_prev_3gcr_culture',
                             prev_swab_col='hx_prev_3gcr_rectal_swab',
                             **kwargs):
    """
    Computes the Holmgren score (2020) for 3GCR Enterobacterales bacteraemia.

    The Holmgren score is an easy-to-use clinical risk-prediction tool designed to
    identify patients at high risk for third-generation cephalosporin-resistant (3GCR)
    Enterobacterales bacteremia, particularly in low-resistance settings. It evaluates
    specific epidemiological and historical markers—such as receiving hospital care abroad,
    previous 3GCR cultures, or prior 3GCR rectal swabs—to assist clinicians in optimizing
    initial antibiotic choices

    Interpretation: Score >= 1 is considered "High Risk" in low-resistance settings.

    !!! warning "Binary Inputs Required"
        This function strictly expects **binary flags (1 or 0)** for all parameters.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        | Clinical Variable                    | Condition Evaluated                    | Points |
        | :----------------------------------- | :------------------------------------- | :----: |
        | Hospital care abroad                 | Hospitalized abroad in last 12 months  |   +1   |
        | Previous 3GCR Culture                | Previous 3GCR in blood or urine        |   +1   |
        | Previous 3GCR Rectal Swab            | Previous 3GCR in rectal swab           |   +1   |

        **References:** Holmgren et al., "An easy-to-use scoring system...", 2020.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    hosp_abroad_col : str, default='hx_hosp_abroad_12m'
        Column name for hospitalization abroad in last 12 months.
    prev_culture_col : str, default='hx_prev_3gcr_culture'
        Column name for previous 3GCR culture.
    prev_swab_col : str, default='hx_prev_3gcr_rectal_swab'
        Column name for previous 3GCR rectal swab.
    **kwargs
        Additional keyword arguments for logging options.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient.
    """
    pass