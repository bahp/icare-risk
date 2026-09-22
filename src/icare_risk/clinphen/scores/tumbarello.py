# Define base weights
TUMBARELLO_WEIGHTS = {
    "tumbarello_hx_prior_esbl": 4,
    "tumbarello_hx_recent_hosp_90d": 2,
    "tumbarello_hx_recent_abx_90d": 2,
    "tumbarello_hx_urinary_catheter": 1
}

TUMBARELLO_HIERARCHY = {}


def calculate_tumbarello_score(df,
                               prior_esbl_col='hx_prior_esbl_any',
                               hosp_90d_col='hx_hosp_last_90d',
                               abx_90d_col='hx_prior_abx_90d',
                               urinary_catheter_col='hx_urinary_catheter_present',
                               **kwargs):
    """
    Computes the Tumbarello/Utrecht-Stockholm ESBL Risk Score.

    The Tumbarello score is a validated risk prediction tool designed specifically to
    identify community-onset bloodstream infections caused by extended-spectrum beta-lactamase
    (ESBL)-producing Enterobacteriaceae. By evaluating risk factors such as prior ESBL
    colonization, recent hospitalizations, prior antibiotic exposure, and the presence
    of a urinary catheter, it stratifies patients upon emergency presentation to help
    guide appropriate empirical therapy.

    Interpretation: High risk is typically defined as a score >= 3 or 4.

    !!! warning "Binary Inputs Required"
        This function strictly expects **binary flags (1 or 0)** for all parameters.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        This specific model is designed for community-onset sepsis, making it a vital
        baseline for patients arriving at the Emergency Department before hospital-acquired
        factors come into play.

        | Clinical Variable           | Condition Evaluated                       | Points |
        | :-------------------------- | :---------------------------------------- | :----: |
        | Prior ESBL                  | Known colonization/infection (any time)   |   +4   |
        | Recent Hospitalization      | Hospitalized within last 90 days          |   +2   |
        | Recent Antibiotics          | Beta-lactams/Quinolones within 90 days    |   +2   |
        | Urinary Catheter            | Permanent or recent urinary catheter      |   +1   |

        **References:** Int J Antimicrob Agents, 2019 (Utrecht/Stockholm Cohort)

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    prior_esbl_col : str, default='hx_prior_esbl_any'
        Column name for prior ESBL history.
    hosp_90d_col : str, default='hx_hosp_last_90d'
        Column name for hospitalization in the last 90 days.
    abx_90d_col : str, default='hx_prior_abx_90d'
        Column name for prior antibiotics in the last 90 days.
    urinary_catheter_col : str, default='hx_urinary_catheter_present'
        Column name for urinary catheter flag.
    **kwargs
        Additional keyword arguments for logging options.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient.
    """
    pass