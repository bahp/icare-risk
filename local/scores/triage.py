import pandas as pd
import numpy as np

from icare_risk.scores import evaluate_score
from icare_risk.utils import validate_required_columns

def calculate_local_ward_triage_score_raw(df,
                                          age_col='AGE_AT_ADMISSION',
                                          stress_col='local_stress_flag',
                                          **kwargs):
    """
    A raw, imperative implementation of the Local Ward Triage score.

    This function demonstrates the manual mathematical calculation using pure
    Pandas/NumPy without invoking the iCARE `evaluate_score` rule engine.
    It does not support audit logging or automatic column validation.
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


def calculate_local_ward_triage_score(df,
                                      age_col='AGE_AT_ADMISSION',
                                      stress_col='local_stress_flag',
                                      **kwargs):
    """
    Computes the Local Ward Triage risk score using the iCARE rule engine.

    This is the preferred, framework-integrated version of the score. It calculates
    a 0-5 point triage score based on advanced age and the presence of a locally
    defined acute stress phenotype. It supports full clinical audit logging.

    !!! warning "Mixed Input Expectations"
        This function requires **raw age** to evaluate the > 65 threshold internally,
        but the stress phenotype must be a pre-calculated **binary flag (1/0)**.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        | Clinical Variable                    | Condition Evaluated                       | Points |
        | :----------------------------------- | :---------------------------------------- | :----: |
        | Advanced Age                         | Age > 65 years                            |   +2   |
        | Acute Stress                         | Local stress phenotype flag is active (1) |   +3   |

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='AGE_AT_ADMISSION'
        Column name for patient age (numeric).
    stress_col : str, default='local_stress_flag'
        Column name for the local stress phenotype flag (binary 1/0).
    **kwargs
        Additional keyword arguments for the rule engine:
        * `verbose` (bool): Enables audit logging. Defaults to False.
        * `logger` (logging.Logger): Logger instance for the audit trail.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed triage score (0 to 5 integers)
        for each patient, matching the input DataFrame index.
    """

    # 1. Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "LOCAL WARD TRIAGE"

    # 2. Validate inputs to prevent silent under-scoring
    validate_required_columns(df,
        required_cols=[age_col, stress_col],
        score_name=score_name
    )

    # 3. Pure Logic Definition for the rule engine
    rules = [
        {
            'desc': 'Age > 65',
            'col': age_col,
            'condition': df[age_col] > 65,
            'points': 2
        },
        {
            'desc': 'Stress Phenotype Active',
            'col': stress_col,
            'condition': df[stress_col] == 1,
            'points': 3
        }
    ]

    # 4. Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)