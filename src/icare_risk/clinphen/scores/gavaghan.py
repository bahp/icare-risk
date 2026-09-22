import pandas as pd
import numpy as np

from typing import Dict, Optional, List


def calculate_gavaghan_score(df,
                             age_col='AGE_AT_ADMISSION',
                             prior_esbl_col='hx_prior_esbl_365d',
                             nursing_home_col='hx_nursing_home_resident',
                             urinary_catheter_col='hx_urinary_catheter_present',
                             prior_abx_col='hx_prior_fc_abx_90d',
                             **kwargs):
    """
    Computes the Gavaghan et al. (2025) ESBL Risk Score.

    The Gavaghan score is a contemporary, tertiary-setting risk assessment tool developed
    to predict the likelihood of ESBL-producing Enterobacterales bacteremia. By weighting
    clinical parameters like advanced age, long-term care residency, indwelling urinary
    catheters, recent broad-spectrum antibiotic exposure, and prior ESBL history, it helps
    clinicians assess resistance risks and tailor initial treatment protocols.

    !!! warning "Mixed Input Expectations"
        This score requires **raw age** to evaluate the >= 65 threshold internally,
        but all other parameters must be pre-calculated **binary flags (1/0)**.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        The Gavaghan et al. (2025) score is a contemporary tool developed specifically for
        risk assessment of ESBL-producing Enterobacterales bacteremia in a tertiary setting.

        | Clinical Variable                    | Condition Evaluated                       | Points |
        | :----------------------------------- | :---------------------------------------- | :----: |
        | Prior ESBL                           | Any ESBL organism within 365 days         |   +4   |
        | Age                                  | Age >= 65 years                           |   +1   |
        | Nursing Home Resident                | Lives in a long-term care facility        |   +2   |
        | Urinary Catheter                     | Indwelling catheter at presentation       |   +1   |
        | Prior Antibiotics                    | Fluoroquinolone or Cephalosporin (90d)    |   +2   |

        **References:** Gavaghan et al., Antimicrob Steward Healthc Epidemiol, 2025.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='AGE_AT_ADMISSION'
        Column name for patient age.
    prior_esbl_col : str, default='hx_prior_esbl_365d'
        Column name for prior ESBL organism within 365 days.
    nursing_home_col : str, default='hx_nursing_home_resident'
        Column name for nursing home resident flag.
    urinary_catheter_col : str, default='hx_urinary_catheter_present'
        Column name for indwelling urinary catheter flag.
    prior_abx_col : str, default='hx_prior_fc_abx_90d'
        Column name for prior antibiotics within 90 days.
    **kwargs
        Additional keyword arguments for logging options.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient.
    """
    pass