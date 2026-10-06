import pandas as pd
import numpy as np

from typing import Dict, Optional, List
from icare_risk.clinphen.scores.utils import calculate_weighted_points
from icare_risk.clinphen.scores.utils import log_score_audit

# Define base weights
HOLMGREM_WEIGHTS = {
    "hx_prior_abx_30d": 4,
    "hx_prior_fc_abx_90d": 1,
    "hx_prior_hosp_abroad": 1
}

HOLMGREM_HIERARCHY = {}

def compute_holmgren_score(df: pd.DataFrame,
                             weights: Optional[Dict[str, int]] = None,
                             verbose: int = 0):
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

    !!! danger "Data Complexity Warning"
        Hospital care abroad is very difficult to compute reliably from routine electronic
        health record (EHR) data due to fragmented international health information systems
        and the lack of structured cross-border coding.

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
    active_weights = weights if weights is not None else HOLMGREM_WEIGHTS

    score = calculate_weighted_points(df, active_weights)

    log_score_audit(
        df,
        active_weights,
        score,
        title="Holmgren Score Breakdown",
        verbose=verbose,
    )

    return score