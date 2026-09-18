import pandas as pd
import numpy as np

from typing import Dict, Optional, List

# Define base weights
CHARLSON_WEIGHTS = {
    "charlson_hx_mi": 1,
    "charlson_hx_chf": 1,
    "charlson_hx_pvd": 1,
    "charlson_hx_stroke": 1,
    "charlson_hx_dementia": 1,
    "charlson_hx_pulmonary": 1,
    "charlson_hx_rheum": 1,
    "charlson_hx_pud": 1,
    "charlson_hx_liver_mild": 1,
    "charlson_hx_diabetes_uncomp": 1,
    "charlson_hx_hemiplegia": 2,
    "charlson_hx_renal_mod_sev": 2,
    "charlson_hx_diabetes_comp": 2,
    "charlson_hx_cancer_solid": 2,
    "charlson_hx_leukemia": 3,
    "charlson_hx_lymphoma": 6,
    "charlson_hx_liver_mod_sev": 3,
    "charlson_hx_cancer_met": 6,
    "charlson_hx_aids": 6,
    'charlson_hx_hiv': 1
}

CHARLSON_HIERARCHY = {
    "charlson_hx_liver_mod_sev": ["charlson_hx_liver_mild"],
    "charlson_hx_diabetes_comp": ["charlson_hx_diabetes_uncomp"],
    "charlson_hx_cancer_met": ["charlson_hx_cancer_solid"],
    "charlson_hx_aids": ["charlson_hx_hiv"]
}


def apply_mutual_exclusivity(
        df: pd.DataFrame,
        hierarchy_map: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Generic helper function for mutual exclusive overrides. Zeros out minor
    conditions if a severe mutually exclusive condition is met (equals 1).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the clinical feature columns.
    hierarchy_map : Dict[str, List[str]]
        A dictionary where the key is the severe column name and the value is a list
        of milder columns that should be overridden (zeroed out) if the severe column is present.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with mutually exclusive overrides applied.
    """
    calc_df = df.copy()

    for severe_col, mild_cols in hierarchy_map.items():
        if severe_col in calc_df.columns:
            # Check where the severe condition is met (value == 1)
            severe_mask = (calc_df[severe_col] == 1)
            if severe_mask.any():
                for mild_col in mild_cols:
                    if mild_col in calc_df.columns:
                        calc_df.loc[severe_mask, mild_col] = 0

    return calc_df

# ------------------------------------------------------------------------
# Score functions
# ------------------------------------------------------------------------
def _calculate_charlson_comorbidity_points(
        df: pd.DataFrame,
        weights: Optional[Dict[str, int]] = None
) -> pd.Series:
    """Compute Charlson comorbidity points for each row."""
    # Set weights
    weights = weights if weights is not None else CHARLSON_WEIGHTS
    # Identify matching columns
    valid_cols = [col for col in weights if col in df.columns]
    # Handle edge case where zero columns match
    if not valid_cols:
        return pd.Series(0, index=df.index)
    # Coerce to numeric to prevent string/object type errors, and fill NaNs
    sub_df = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Align weights series strictly with the valid columns
    active_weights = pd.Series({col: weights[col] for col in valid_cols})
    # Return
    return sub_df.dot(active_weights)


def _calculate_charlson_age_points(df: pd.DataFrame, age_col: str) -> pd.Series:
    """Pure math: computes age adjustment points vectorially."""
    if age_col not in df.columns:
        return pd.Series(0, index=df.index)

    age = pd.to_numeric(df[age_col], errors='coerce').fillna(0)
    conditions = [
        (age >= 50) & (age <= 59),
        (age >= 60) & (age <= 69),
        (age >= 70) & (age <= 79),
        (age >= 80)
    ]
    choices = [1, 2, 3, 4]
    return pd.Series(np.select(conditions, choices, default=0), index=df.index)


def _log_charlson_audit_trace(
    df: pd.DataFrame,
    calc_df: pd.DataFrame,
    age_col: str,
    weights: Dict[str, int],
    hierarchy_map: Dict[str, List[str]],
    total_score: pd.Series,
    verbose: int
) -> None:
    """Handles all logging, verbosity thresholds, missing columns, and overridden traces for Patient #1."""
    if verbose <= 0:
        return

    print("  [Charlson Comorbidity Index Breakdown]")
    if len(df) > 1:
        print(f"    ⚠️ WARNING: DataFrame contains {len(df)} rows.")
        print(f"    ⚠️ The calculated score applies to all rows, but this trace only details Patient #1 (Index 0).")

    # Detect hierarchy overrides for Patient #1
    overridden_cols = set()
    if len(df) > 0:
        for severe_col, mild_cols in hierarchy_map.items():
            if severe_col in df.columns and df[severe_col].iloc[0] == 1:
                for mild_col in mild_cols:
                    if mild_col in df.columns and df[mild_col].iloc[0] == 1 and calc_df[mild_col].iloc[0] == 0:
                        overridden_cols.add(mild_col)

    # Log Comorbidity Evaluation
    for col, weight in weights.items():
        if col not in df.columns:
            if verbose >= 2:
                print(f"    [?] {col:<32} (Missing Col ) +0")
            continue

        raw_val = df[col].iloc[0] if len(df) > 0 else 0
        is_met = (calc_df[col] == 1).iloc[0] if len(calc_df) > 0 else False
        pts_added = weight if is_met else 0
        is_overridden = col in overridden_cols

        if verbose >= 2:
            if is_overridden:
                print(f"    [-] {col:<32} (Overridden  ) +0")
            elif is_met:
                print(f"    [+] {col:<32} (Value: {str(raw_val):<5}) +{pts_added}")
            else:
                print(f"    [ ] {col:<32} (Value: {str(raw_val):<5}) +0")
        elif verbose == 1 and is_met:
            print(f"    [+] {col:<32} (Value: {str(raw_val):<5}) +{pts_added}")

    # Log Age Evaluation
    if age_col in df.columns:
        age_val = df[age_col].iloc[0] if len(df) > 0 else 0
        age_pts_p1 = _calculate_charlson_age_points(df, age_col).iloc[0] if len(df) > 0 else 0
        print(f"    [+] {f'Age Adjustment ({age_col})':<32} (Value: {str(age_val):<5}) +{age_pts_p1}")
    elif verbose >= 2:
        print(f"    [?] {f'Age Adjustment ({age_col})':<32} (Missing Col) +0")

    final_score_p1 = total_score.iloc[0] if len(total_score) > 0 else 0
    print(f"    {'-' * 55}")
    print(f"    [=] TOTAL COMPUTED SCORE:         {final_score_p1}\n")


def compute_charlson_score(df: pd.DataFrame,
                           age_col: str = "AGE_AT_ADMISSION",
                           weights: Optional[Dict[str, int]] = None,
                           verbose: int = 0) -> pd.Series:
    """
    Computes the Age-Adjusted Charlson Comorbidity Index with hierarchical overrides and audit logging.

    The Charlson Quan score is a refined, widely adopted adaptation of the original
    Charlson Comorbidity Index that utilizes administrative ICD coding algorithms
    developed by Quan et al. It applies optimized comorbidity weights and strict
    hierarchical rules to handle overlapping or progressive conditions (such as
    distinguishing mild versus severe liver disease or complicated versus uncomplicated
    diabetes) to yield a more accurate prognostic risk score using routine electronic
    health record data.

    !!! warning "Hierarchical Logic & Binary Inputs"
        This function evaluates hierarchy. For example, if a patient has both
        mild liver disease and severe liver disease, only the severe points are
        awarded. Requires **binary (1/0)** flags for conditions and **raw age**.

    ??? note "Clinical Logic & Point Allocation (Click to expand)"
        ```yaml
        --8<-- "src/icare_risk/config/feature_config.yaml:charlson_quan_config"
        ```

        * **Age Adjustment:** 50-59 (+1), 60-69 (+2), 70-79 (+3), 80+ (+4)
        * **1-Point Conditions:** MI, CHF, PVD, Stroke, Dementia, Pulmonary, Rheum, PUD.
        * **Hierarchical Categories (Highest weight wins):**
            * Liver: Severe (+3), Mild (+1)
            * Diabetes: Complicated (+2), Uncomplicated (+1)
            * Cancer: Metastatic (+6), Solid Tumor (+2)
            * HIV/AIDS: AIDS (+6), HIV (+1)
        * **Other Severe:** Renal Moderate/Severe (+2)

        **References:** Quan, H. et al. Coding algorithms for defining comorbidities in
        ICD-9-CM and ICD-10 administrative data. Medical Care, 43(11), 1130–1139. (2005).

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing binary comorbidity flags and age data.
    age_col : str, optional
        Column name representing raw patient age, by default "AGE_AT_ADMISSION".
    weights : Dict[str, int], optional
        Custom dictionary mapping comorbidity columns to integer weights.
        If None, defaults to `CHARLSON_WEIGHTS`.
    verbose : int, optional
        Verbosity level for audit logging (0 = silent, 1 = active scores only, 2 = full diagnostic trace).
        Default is 0.

    Returns
    -------
    pd.Series
        A pandas Series containing the final computed Charlson score for each encounter.
    """
    # Libraries
    from icare_risk.clinphen.utils.validation import validate_required_columns

    active_weights = weights if weights is not None else CHARLSON_WEIGHTS

    # 1. Validate required columns
    required_cols = list(active_weights.keys()) + [age_col]
    validate_required_columns(df, required_cols,
        score_name="Charlson Comorbidity Index", verbose=(verbose > 0))

    # 2. Apply business rules (mutual exclusivity overrides)
    calc_df = apply_mutual_exclusivity(df, CHARLSON_HIERARCHY)

    # 3. Compute vectorized components
    comorbidity_scores = _calculate_charlson_comorbidity_points(calc_df, active_weights)
    age_scores = _calculate_charlson_age_points(df, age_col)
    total_score = comorbidity_scores + age_scores

    # 4. Generate debugging/audit trace if requested
    _log_charlson_audit_trace(df,
        calc_df=calc_df, age_col=age_col, weights=active_weights,
        hierarchy_map=CHARLSON_HIERARCHY, total_score=total_score,
        verbose=verbose)

    # Return
    return total_score