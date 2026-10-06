import pandas as pd

from typing import Dict, Optional
from icare_risk.clinphen.scores.utils import calculate_weighted_points
from icare_risk.clinphen.scores.utils import log_score_audit

# Define base weights
KIM_WEIGHTS = {
    "kim_hx_prior_esbl": 5,
    "kim_hx_recent_hosp_1yr": 2,
    "kim_hx_nursing_home_resident": 2,
    "kim_hx_urinary_catheter": 1,
    "kim_hx_prior_antibiotics_90d": 1
}

JONES_HIERARCHY = {}


def compute_kim_score(df: pd.DataFrame,
        weights: Optional[Dict[str, int]] = None,
        verbose: int = 0,
    ) -> pd.Series:
    """
    Computes the Kim et al. (2019) ESBL Risk Score.

    !!! warning "Binary Inputs Required"
        This function strictly expects **binary flags (1 or 0)** for all parameters.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        It focuses on identifying risk factors specifically for community-onset BSIs caused
        by ESBL-producing E. coli and Klebsiella species. This model is particularly useful
        for differentiating resistant from susceptible strains right at the point of admission.

        | Clinical Variable           | Condition Evaluated                       | Points |
        | :-------------------------- | :---------------------------------------- | :----: |
        | Prior ESBL                  | Prior ESBL colonization or infection      |   +5   |
        | Recent Hospitalization      | Hospitalization within the last 1 year    |   +2   |
        | Nursing Home Resident       | Resident in a long-term care facility     |   +2   |
        | Urinary Catheter            | Use of indwelling urinary catheter        |   +1   |
        | Prior Antibiotics           | Use of antibiotics within 90 days         |   +1   |

        **References:** Kim et al., J Korean Med Sci, 2019

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    prior_esbl_col : str, default='hx_prior_esbl_any'
        Column name for prior ESBL history.
    hosp_1y_col : str, default='hx_hosp_last_365d'
        Column name for hospitalization in the last 1 year.
    nursing_home_col : str, default='hx_nursing_home_resident'
        Column name for nursing home resident flag.
    urinary_catheter_col : str, default='hx_urinary_catheter_present'
        Column name for urinary catheter use.
    prior_abx_90d_col : str, default='hx_prior_abx_90d'
        Column name for antibiotic use within 90 days.
    **kwargs
        Additional keyword arguments for logging options.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient
    """
    active_weights = weights if weights is not None else KIM_WEIGHTS

    score = calculate_weighted_points(df, active_weights)

    log_score_audit(
        df,
        active_weights,
        score,
        title="Kim Score Breakdown",
        verbose=verbose,
    )

    return score
