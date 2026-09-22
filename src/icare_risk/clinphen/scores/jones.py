
# Define base weights
JONES_WEIGHTS = {
    "jones_hx_prior_esbl_180d": 5,
    "jones_hx_prior_antibiotics_30d": 2,
    "jones_hx_chronic_dialysis": 2,
    "ones_hx_transfer_from_hospital": 1
}

JONES_HIERARCHY = {}

def calculate_jones_score(df,
                          prior_esbl_col='hx_prior_esbl_180d',
                          prior_abx_col='hx_prior_abx_30d',
                          chronic_dialysis_col='hx_chronic_dialysis',
                          transfer_hosp_col='hx_transfer_from_hosp',
                          **kwargs):
    """
    Computes the Jones et al. (2025) ESBL Risk Score for Non-Urinary Isolates.

    The Jones score is a specialized clinical risk-prediction tool designed specifically
    to assess the likelihood of extended-spectrum beta-lactamase (ESBL)-producing infections
    from non-urinary isolates (such as bloodstream or respiratory sources). Because risk
    factors for resistant pathogens can differ significantly between localized urinary tracts
    and systemic sites, this score helps clinicians evaluate high-risk markers—such as prior
    ESBL history, recent antibiotic exposure, chronic dialysis, and hospital transfers—right
    at the point of care.

    !!! warning "Binary Inputs Required"
        This function strictly expects **binary flags (1 or 0)** for all parameters.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        This score is a specialized tool designed specifically for non-urinary isolates.
        This is a crucial distinction in clinical practice, as risk factors for ESBL in
        bloodstream or respiratory infections often differ from those in simple UTIs.

        | Clinical Variable           | Condition Evaluated                       | Points |
        | :-------------------------- | :---------------------------------------- | :----: |
        | Prior ESBL                  | Positive ESBL culture within 180 days     |   +5   |
        | Prior Antibiotics           | Any antibiotic use within 30 days         |   +2   |
        | Chronic Dialysis            | Patient on hemodialysis or peritoneal     |   +2   |
        | Transfer from Hospital      | Admission via transfer from another hosp  |   +1   |

        **References:** Jones et al., Pharmacotherapy, 2025.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    prior_esbl_col : str, default='hx_prior_esbl_180d'
        Column name for prior ESBL within 180 days.
    prior_abx_col : str, default='hx_prior_abx_30d'
        Column name for prior antibiotic use within 30 days.
    chronic_dialysis_col : str, default='hx_chronic_dialysis'
        Column name for chronic dialysis flag.
    transfer_hosp_col : str, default='hx_transfer_from_hosp'
        Column name for transfer from another hospital flag.
    **kwargs
        Additional keyword arguments for logging options.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient.
    """
    pass