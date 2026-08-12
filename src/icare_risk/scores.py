"""
Author:
Date:

This module contains the pure mathematical logic for calculating clinical risk
scores based on the clean, standardized phenotypes extracted in `phenotypes.py`.

SCORE CLASSIFICATIONS
=====================
To understand the data pipeline, you must distinguish between the two distinct
tiers of clinical scores calculated in this file:

1. INTERMEDIATE (PARTIAL) SCORES
   These scores summarize a patient's baseline health or acute illness severity
   into a single number, acting as "super-features" that reduce noise.
   - Purpose: To serve as building blocks for more complex final models.
   - Examples: `calculate_charlson_score()`, `calculate_pitt_bacteremia_score()`

2. FINAL (PREDICTIVE) SCORES
   These are specialized, extensively validated algorithms designed to answer
   one specific clinical question or predict a specific outcome (e.g., mortality).
   - Purpose: To act as the ultimate decision-support endpoint, often consuming
     intermediate scores as inputs.
   - Examples: `calculate_increment_esbl()`, `calculate_gavaghan_score()`

DESIGN RULES
============
To maintain a clean and testable architecture, adhere strictly to these rules:

1. NO DATA EXTRACTION: Functions in this file should ONLY contain mathematical
   logic (e.g., `np.where`, addition, conditionals). Do not search for ICD-10
   codes, clinical keywords, or evaluate messy string lab reports here. All
   messy data translation MUST happen upstream in `phenotypes.py`.

2. AUDIT LOGGING: Use the provided `evaluate_score` rule engine or `audit_log`
   helpers to ensure every calculated point can be traced back and verified
   by a clinician debugging the score.
"""

# src/scores.py
import pandas as pd
import numpy as np

from icare_risk.utils import validate_required_columns

# ------------------------------------------------------------------------
# Helper methods
# ------------------------------------------------------------------------
def audit_log(condition, points, description, verbose=False, logger=None):
    """
    Updated observer helper: Logs the status of every clinical element.
    """
    if verbose and logger:
        # Evaluate if the condition is met (handles pandas Series and numpy arrays)
        is_met = condition.iloc[0] if hasattr(condition, 'iloc') else bool(np.array(condition)[0])
        status_icon = "[+]" if is_met else "[ ]"
        added_points = points if is_met else 0              +0
        logger.info(f"    {status_icon} {description:<35} +{added_points}")



def evaluate_score(df, rules, score_name, verbose=False, logger=None):
    """
    A rule engine that calculates the total score and automatically
    generates a clean debugging trace if requested.
    """
    total_score = pd.Series(0, index=df.index)

    # --- DEBUG HEADER & WARNING ---
    if verbose and logger:
        logger.info(f"  [{score_name} Breakdown]")

        # Check if the user passed a batch instead of a single patient
        if len(df) > 1:
            logger.warning(f"    ⚠️ WARNING: DataFrame contains {len(df)} rows.")
            logger.warning(
                f"    ⚠️ The calculated score applies to all rows, but this trace only shows Patient #1 (Index 0).")

    # --- RULE EVALUATION ---
    for rule in rules:
        col = rule['col']
        points = rule['points']
        desc = rule['desc']

        # Safety check: skip if column is missing
        if col not in df.columns:
            if verbose and logger:
                logger.info(f"    [?] {desc:<28} (Missing Col) +0")
            continue

        # 1. Math/Logic: Calculate and add points
        condition = rule['condition']
        pts_awarded = np.where(condition, points, 0)
        total_score += pts_awarded

        # 2. Debugging: Log the evaluation
        if verbose and logger:
            # Extract single values for the log (assuming case-by-case processing)
            is_met = bool(np.array(condition)[0])
            raw_val = df[col].iloc[0]
            pts_added = points if is_met else 0
            icon = "[+]" if is_met else "[ ]"

            # Format: [+] Age > 50 (Value: 72) +3
            logger.info(f"    {icon} {desc:<28} (Value: {raw_val:<5}) +{pts_added}")

    if verbose and logger:
        logger.info(f"    {'-' * 45}")
        logger.info(f"    [=] TOTAL COMPUTED:            {total_score.iloc[0]}\n")

    return total_score



# --------------------------------------------------------------------------------
#                   Intermediate clinical scores
# --------------------------------------------------------------------------------
# Intermediate clinical scores act as a bridge between raw, messy patient data and
# the final predictive model. Instead of evaluating dozens of individual, noisy
# variables (like a specific ICD-10 code, a scattered lab value, or a fluctuating
# heart rate) in isolation, we aggregate them into validated, standardized metrics.
#
# Scores such as the Charlson Comorbidity Index (CCI), the Pitt Bacteremia Score, or
# SIRS criteria summarize a patient's chronic health baseline or acute illness severity
# into a single number. In a data science pipeline, these intermediate metrics act as
# clinically interpretable "super-features" that reduce noise and provide a stable
# foundation for computing the final risk scores.

def calculate_mews(df, **kwargs):
    """Computes the Modified Early Warning Score (MEWS).

    The Modified Early Warning Score (MEWS) is a standardized clinical tool used
    to quickly identify hospitalized patients who are at risk of sudden, severe
    deterioration. By assigning points to vital signs that deviate from normal
    physiological ranges, it provides clinicians with an objective metric that
    can automatically trigger a rapid response team or escalated care before a
    major adverse event occurs.

    !!! warning "Raw Vitals Required"
        Unlike intermediate scores that take binary flags, this function requires
        **raw, continuous vital signs** (e.g., HR of 115, RR of 22). It internally
        maps these continuous values to clinical derangement points.

    ??? note "Clinical Logic & Point Allocation (Click to expand)"
        | Clinical Variable | +1 Point | +2 Points | +3 Points |
        | :--- | :--- | :--- | :--- |
        | **Respiratory Rate (RR)** | 15–20 | <=8 or 21–29 | >=30 |
        | **Heart Rate (HR)** | 41–50 or 101–110 | <=40 or 111–129 | >=130 |
        | **Systolic BP (SBP)** | 81–100 | 71–80 or >=200 | <=70 |
        | **Temperature (°C)** | - | <35 or >38.5 | - |

    Parameters
    ----------
    df : pandas.DataFrame
        The patient dataframe containing raw vitals.
    **kwargs
        Arbitrary keyword arguments mapping clinical variables to column names:
        * `rr_col` (str): Defaults to 'rr'.
        * `hr_col` (str): Defaults to 'hr'.
        * `sbp_col` (str): Defaults to 'sbp'.
        * `temp_col` (str): Defaults to 'temp'.
        * `verbose` (bool): Enables audit logging. Defaults to False.
        * `logger` (logging.Logger): Logger instance for the audit trail.

    Returns
    -------
    pandas.Series
        A pandas Series containing the computed MEWS score for each patient.
    """
    score_name = "MEWS Score"
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)

    # Extract column names from YAML
    rr_col = kwargs.get('rr_col', 'rr')
    hr_col = kwargs.get('hr_col', 'hr')
    sbp_col = kwargs.get('sbp_col', 'sbp')
    temp_col = kwargs.get('temp_col', 'temp')

    rules = []

    # 1. Respiratory Rate (RR) Logic
    if rr_col in df.columns:
        rr = pd.to_numeric(df[rr_col], errors='coerce')
        rr_pts = pd.Series(0, index=df.index)
        rr_pts.loc[rr <= 8] = 2
        rr_pts.loc[(rr >= 15) & (rr <= 20)] = 1
        rr_pts.loc[(rr >= 21) & (rr <= 29)] = 2
        rr_pts.loc[rr >= 30] = 3
        rules.append({'desc': 'RR Derangement',
                      'col': rr_col,
                      'condition': rr_pts > 0,
                      'points': rr_pts})

    # 2. Heart Rate (HR) Logic
    if hr_col in df.columns:
        hr = pd.to_numeric(df[hr_col], errors='coerce')
        hr_pts = pd.Series(0, index=df.index)
        hr_pts.loc[hr <= 40] = 2
        hr_pts.loc[(hr >= 41) & (hr <= 50)] = 1
        hr_pts.loc[(hr >= 101) & (hr <= 110)] = 1
        hr_pts.loc[(hr >= 111) & (hr <= 129)] = 2
        hr_pts.loc[hr >= 130] = 3
        rules.append({'desc': 'HR Derangement',
                      'col': hr_col,
                      'condition': hr_pts > 0,
                      'points': hr_pts})

    # 3. Systolic BP (SBP) Logic
    if sbp_col in df.columns:
        sbp = pd.to_numeric(df[sbp_col], errors='coerce')
        sbp_pts = pd.Series(0, index=df.index)
        sbp_pts.loc[sbp <= 70] = 3
        sbp_pts.loc[(sbp >= 71) & (sbp <= 80)] = 2
        sbp_pts.loc[(sbp >= 81) & (sbp <= 100)] = 1
        sbp_pts.loc[sbp >= 200] = 2
        rules.append({'desc': 'SBP Derangement',
                      'col': sbp_col,
                      'condition': sbp_pts > 0,
                      'points': sbp_pts})

    # 4. Temperature Logic
    if temp_col in df.columns:
        temp = pd.to_numeric(df[temp_col], errors='coerce')
        temp_pts = pd.Series(0, index=df.index)
        temp_pts.loc[temp < 35] = 2
        temp_pts.loc[temp > 38.5] = 2
        rules.append({'desc': 'Temp Derangement',
                      'col': temp_col,
                      'condition': temp_pts > 0,
                      'points': temp_pts})

    return evaluate_score(df, rules, score_name, verbose, logger)


def calculate_charlson(df, **kwargs):
    """Computes the Age-Adjusted Charlson Comorbidity Index (CCI).

    The Charlson Comorbidity Index is a widely validated, foundational clinical tool
    used to categorize a patient's comorbid health conditions and predict short- and
    long-term mortality. By combining weighted scores for various chronic diseases
    (such as cardiovascular disease, diabetes, and organ failure) with an age-adjustment
    factor, it provides a comprehensive baseline measure of a patient's underlying
    chronic health burden.

    !!! warning "Mixed Input Expectations"
        This function requires a **raw continuous age** column alongside a dictionary
        of **binary (1/0) flags** for comorbidities. Do not pass raw ICD-10 codes.

    ??? note "Clinical Logic & Point Allocation (Click to expand)"
        You can inspect the exact default YAML rule mappings used to power this function
        in the project configuration. The relevant content from that file
        (config/feature_config.yml) is shown below.

        ```yaml
        --8<-- "src/icare_risk/config/feature_config.yaml:charlson_config"
        ```

        | Weight | Comorbidities Evaluated |
        | :---: | :--- |
        | **1 Point** | Myocardial Infarction, Congestive Heart Failure, Peripheral Vascular Disease, Cerebrovascular Disease, Dementia, Chronic Pulmonary Disease, Connective Tissue Disease, Peptic Ulcer Disease, Mild Liver Disease, Uncomplicated Diabetes |
        | **2 Points** | Hemiplegia or Paraplegia, Moderate to Severe Renal Disease, Diabetes with Chronic Complications, Non-Metastatic Solid Tumor, Leukemia, Lymphoma |
        | **3 Points** | Moderate to Severe Liver Disease |
        | **6 Points** | Metastatic Solid Tumor, AIDS / HIV |

        * **Age Adjustment:** 50-59 (+1), 60-69 (+2), 70-79 (+3), 80+ (+4)

        **References:** Charlson, M. E., et al. A new method of
        classifying prognostic comorbidity in longitudinal studies: Development and validation.
        Journal of Chronic Diseases, 40(5), 373–383. (1987)

    Parameters
    ----------
    df : pandas.DataFrame
        The patient dataframe.
    **kwargs
        Arbitrary keyword arguments containing:
        * `age_col` (str): Column name for raw patient age. Defaults to 'AGE_AT_ADMISSION'.
        * `comorbidities` (dict): A dictionary mapping column names to their clinical weights.

    Returns
    -------
    pandas.Series
        A pandas Series containing the computed CCI score for each patient.
    """
    score_name = "Charlson Comorbidity Index"
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)

    # 1. Dynamic Comorbidity Rules
    # Maps YAML column names to their clinical weights
    comorbidities = kwargs.get('comorbidities', {})
    rules = []

    for col, weight in comorbidities.items():
        if col in df.columns:
            rules.append({
                'desc': f'Comorbidity: {col}',
                'col': col,
                'condition': df[col] == 1,
                'points': weight
            })

    # 2. Age-Adjusted Component (+1 point for every decade over 40)
    age_col = kwargs.get('age_col', 'AGE_AT_ADMISSION')
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors='coerce')
        # Logic: (Age - 40) // 10 + 1 for the 5th decade (50s)
        # Simplified: if age=50 (1pt), 60 (2pts), 70 (3pts), 80 (4pts)
        age_pts = np.maximum(0, (age - 40) // 10)

        # We manually add age points as they aren't a simple binary flag
        # but we can still register a 'dummy' rule for the audit log
        rules.append({
            'desc': 'Age Adjustment (>40)',
            'col': age_col,
            'condition': age > 40,
            'points': age_pts
        })

    # 3. Execution via shared engine
    return evaluate_score(df, rules, score_name, verbose, logger)



def calculate_charlson_quan(df, **kwargs):
    """Computes the Charlson Comorbidity Index using Quan et al. hierarchical weights.

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
    df : pandas.DataFrame
        The patient dataframe.
    **kwargs
        Keyword arguments mapping clinical conditions to binary column names (e.g.,
        `age_col`, `mi_col`, `liver_sev_col`, `cancer_met_col`).

    Returns
    -------
    numpy.ndarray
        An array of computed Charlson Quan scores aligned with `df.index`.
    """
    score = pd.Series(0, index=df.index)

    # 1. Age Component
    age_col = kwargs.get('age_col', 'AGE_AT_ADMISSION')
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors='coerce')
        score += np.where((age >= 50) & (age <= 59), 1, 0)
        score += np.where((age >= 60) & (age <= 69), 2, 0)
        score += np.where((age >= 70) & (age <= 79), 3, 0)
        score += np.where(age >= 80, 4, 0)

    # 2. Apply Simple 1-Point Categories
    simple_cols = [
        kwargs.get('mi_col'), kwargs.get('chf_col'), kwargs.get('pvd_col'),
        kwargs.get('stroke_col'), kwargs.get('dementia_col'),
        kwargs.get('pulmonary_col'), kwargs.get('rheum_col'), kwargs.get('pud_col')
    ]

    for col in simple_cols:
        # Check if the col name was provided AND exists in the dataframe
        if col and col in df.columns:
            score += df[col].fillna(0).astype(int)

    # 3. Apply Hierarchical Categories (Higher weight trumps lower)
    # Get column names from kwargs
    liver_sev = kwargs.get('liver_sev_col')
    liver_mild = kwargs.get('liver_mild_col')
    diab_comp = kwargs.get('diabetes_comp_col')
    diab_uncomp = kwargs.get('diabetes_uncomp_col')
    canc_met = kwargs.get('cancer_met_col')
    canc_solid = kwargs.get('cancer_solid_col')
    aids_col = kwargs.get('aids_col')
    hiv_col = kwargs.get('hiv_col')
    renal_sev = kwargs.get('renal_sev_col')

    # Liver (Severe=3, Mild=1)
    if liver_sev in df.columns:
        mild_pts = df.get(liver_mild, pd.Series(0, index=df.index)).fillna(0)
        score += np.where(df[liver_sev] == 1, 3, mild_pts)

    # Diabetes (Complicated=2, Uncomplicated=1)
    if diab_comp in df.columns:
        uncomp_pts = df.get(diab_uncomp, pd.Series(0, index=df.index)).fillna(0)
        score += np.where(df[diab_comp] == 1, 2, uncomp_pts)

    # Cancer (Metastatic=6, Solid=2)
    if canc_met in df.columns:
        solid_flag = df.get(canc_solid, pd.Series(0, index=df.index)).fillna(0)
        score += np.where(df[canc_met] == 1, 6, (solid_flag * 2))

    # HIV/AIDS (AIDS=6, HIV=1)
    if aids_col in df.columns:
        hiv_pts = df.get(hiv_col, pd.Series(0, index=df.index)).fillna(0)
        score += np.where(df[aids_col] == 1, 6, hiv_pts)

    # Renal (Moderate/Severe=2)
    if renal_sev in df.columns:
        score += np.where(df[renal_sev] == 1, 2, 0)

    return score.values


def calculate_pitt_score(df, **kwargs):
    """Computes the Pitt Bacteremia Score (0 to 14 points).

    The Pitt Bacteremia Score is a validated clinical instrument used to measure acute
    illness severity in patients experiencing bloodstream infections. By evaluating
    parameters such as mental status, fever status, hypotension, mechanical ventilation,
    and recent cardiac arrest, it assigns a discrete point score (typically ranging from
    0 to 14) that helps clinicians estimate short-term mortality risk and guide urgent
    therapeutic decisions.

    !!! warning "Pre-computed Inputs Required"
        This function expects **pre-calculated points** and binary flags, NOT raw clinical values.
        For example, `temp_col` must contain the discrete Pitt points (0, 1, or 2), not the raw
        temperature in Celsius. Passing raw vitals will result in massive calculation errors.

    ??? note "Clinical Logic & Point Allocation (Click to expand)"
        Because 'temp' and 'mental' phenotypes already return the exact points (0, 1, 2, 4),
        we multiply them by 1. The boolean flags get multiplied by their specific Pitt weights.

        * **Fever Status:** +1 or +2 points (Pre-computed input required)
        * **Mental Status:** +1, +2, or +4 points (Pre-computed input required)
        * **Hypotension:** +2 points (Requires binary 1/0 flag)
        * **Mechanical Ventilation:** +2 points (Requires binary 1/0 flag)
        * **Cardiac Arrest:** +4 points (Requires binary 1/0 flag)

        Missing columns are safely handled and default to 0 points.

        **References:**  Paterson, D. L. et al International prospective study of Klebsiella pneumoniae
        bacteremia: implications of extended-spectrum beta-lactamase production in nosocomial infections.
        Annals of Internal Medicine, 140(1), 26–32. (2004)

    Parameters
    ----------
    df : pandas.DataFrame
        The patient dataframe.
    **kwargs
        Arbitrary keyword arguments mapping clinical variables to column names:
        * `temp_col` (str): Defaults to 'pitt_fever_status_score'.
        * `mental_col` (str): Defaults to 'pitt_mental_status_score'.
        * `hypotens_col` (str): Defaults to 'pitt_hypotension_flag'.
        * `vent_col` (str): Defaults to 'pitt_mech_vent_flag'.
        * `arrest_col` (str): Defaults to 'pitt_cardiac_arrest_flag'.

    Returns
    -------
    numpy.ndarray
        An array of computed Pitt scores aligned with `df.index`.
    """
    score = pd.Series(0, index=df.index)

    # Map the expected columns to their point multipliers.
    # Because 'temp' and 'mental' phenotypes already return the exact points (0, 1, 2, 4),
    # we just multiply them by 1. The boolean flags get multiplied by their Pitt weights.
    components = {
        kwargs.get('temp_col', 'pitt_fever_status_score'): 1,
        kwargs.get('mental_col', 'pitt_mental_status_score'): 1,
        kwargs.get('hypotens_col', 'pitt_hypotension_flag'): 2,
        kwargs.get('vent_col', 'pitt_mech_vent_flag'): 2,
        kwargs.get('arrest_col', 'pitt_cardiac_arrest_flag'): 4
    }

    # Iterate and add, safely handling missing data
    for col, weight in components.items():
        if col in df.columns:
            score += df[col].fillna(0) * weight

    return score.values


def calculate_sirs(df, **kwargs):
    """
    Computes the number of SIRS criteria met (0 to 4).

    The Systemic Inflammatory Response Syndrome (SIRS) score is a clinical screening
    tool used to identify generalized systemic inflammation and severe stress responses
    to conditions like infection, trauma, or burns. By evaluating four physiological
    parameters—heart rate, respiratory rate, core body temperature, and white blood
    cell count—it measures whether a patient meets criteria for widespread inflammatory
    activation, which historically served as a foundational framework for defining sepsis.

    !!! warning "Binary Inputs Required"
        This function strictly requires **binary flags (1 or 0)** for each of the four
        SIRS criteria. It does NOT evaluate raw vital signs (like heart rate > 90).
        Raw data must be translated into binary flags upstream in `phenotypes.py`.

    ??? note "Calculation Logic (Click to expand)"
        Simply adds up the pre-computed 1/0 flags for Tachycardia, Tachypnea,
        Abnormal Temperature, and Abnormal WBC. Missing data is treated as 0
        (condition not met).

        **References:** Bone, R. C. (1992). Definitions for sepsis and organ failure
        and guidelines for the use of innovative therapies in sepsis. Chest, 101(6),
        1644–1655.

    Parameters
    ----------
    df : pandas.DataFrame
        The patient dataframe.
    **kwargs
        Arbitrary keyword arguments mapping clinical variables to column names:
        * `tachycardia_col` (str): Defaults to 'sirs_tachycardia_flag'.
        * `tachypnea_col` (str): Defaults to 'sirs_tachypnea_flag'.
        * `temp_col` (str): Defaults to 'sirs_abnormal_temp_flag'.
        * `wbc_col` (str): Defaults to 'sirs_abnormal_wbc_flag'.

    Returns
    -------
    numpy.ndarray
        An array of computed SIRS counts (0-4) aligned with `df.index`.
    """
    score = pd.Series(0, index=df.index)

    # Get the column names mapped in the YAML
    cols_to_add = [
        kwargs.get('tachycardia_col', 'sirs_tachycardia_flag'),
        kwargs.get('tachypnea_col', 'sirs_tachypnea_flag'),
        kwargs.get('temp_col', 'sirs_abnormal_temp_flag'),
        kwargs.get('wbc_col', 'sirs_abnormal_wbc_flag')
    ]

    # Add them up, treating missing data as 0 (condition not met)
    for col in cols_to_add:
        if col in df.columns:
            score += df[col].fillna(0)

    return score.values



# --------------------------------------------------------------------------------
#                            Main clinical scores
# --------------------------------------------------------------------------------
# Main clinical scores represent the ultimate endpoint of the phenotyping and feature
# engineering pipeline. These are specialized, extensively validated algorithms designed
# to answer a specific clinical question or predict a precise outcome—such as 30-day
# mortality or the likelihood of an ESBL-producing infection.
#
# By synthesizing direct clinical variables (e.g., patient age, source of infection)
# with intermediate clinical scores (e.g., Pitt Bacteremia Score, CCI), the final score
# stratifies patients into actionable risk categories. In this pipeline, tools like the
# INCREMENT-ESBL, Gavaghan, or Jones scores serve as decision-support endpoints, directly
# guiding clinical interventions like escalating or de-escalating empirical antibiotic
# therapy.


def calculate_increment_esbl_v2(df, age_col='age',
                             charlson_col='charlson_score',
                             pitt_col='pitt_score',
                             sirs_col='sirs_count',
                             bsi_source_col='bsi_source',
                             microorganism_col='microorganism',
                             inapprop_abx_col='inappropriate_abx'):
    """
    Computes the INCREMENT-ESBL predictive score for mortality.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='age'
        Column name for patient age (numeric).
    charlson_col : str, default='charlson_score'
        Column name for Charlson comorbidity score (numeric).
    pitt_col : str, default='pitt_score'
        Column name for Pitt bacteremia score (numeric).
    sirs_col : str, default='sirs_count'
        Column name for SIRS criteria count (numeric).
    bsi_source_col : str, default='bsi_source'
        Column name for infection source (string).
    microorganism_col : str, default='microorganism'
        Column name for isolated organism (string).
    inapprop_abx_col : str, default='inappropriate_abx'
        Column name for inappropriate antibiotic therapy flag (binary 1/0).

    Returns
    -------
    pd.Series
        A pandas Series containing the computed INCREMENT-ESBL score (integers)
        for each patient, matching the input DataFrame index.

    Notes
    -----
    **Clinical Criteria & Point Allocation**

    | Clinical Variable                      | Condition Evaluated               | Points |
    | :------------------------------------- | :-------------------------------- | :----: |
    | Demographics                           | Age > 50 years                    |   +3   |
    | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
    | Acute Underlying Severity              | High Pitt bacteremia score (>=6)  |   +3   |
    | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
    | SIRS Severity                          | Severe SIRS / Shock present       |   +4   |
    | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
    | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

    References
    ----------
    Palacios-Baena Z, et al. Development and validation of the INCREMENT-ESBL
    predictive score for mortality in patients with bloodstream infections.
    J Antimicrob Chemother. 2017;72(3):906-913.
    """


    score = pd.Series(0, index=df.index)

    # Demographics
    if age_col in df.columns:
        score += np.where(df[age_col] > 50, 3, 0)

    # Severe Comorbidities (Defined by paper as Charlson > 3)
    if charlson_col in df.columns:
        score += np.where(df[charlson_col] > 3, 4, 0)

    # Acute Severity (Defined by paper as Pitt score >= 6)
    if pitt_col in df.columns:
        score += np.where(df[pitt_col] >= 6, 3, 0)

    # Severe SIRS / Shock (Defined as >= 2 SIRS criteria + hypotension/shock)
    if sirs_col in df.columns:
        score += np.where(df[sirs_col] >= 2, 4, 0)  # Simplified for example

    # BSI Source
    if bsi_source_col in df.columns:
        is_urinary = df[bsi_source_col].astype(str) \
            .str.lower().str.contains('urinary|uti', na=False)
        score += np.where(~is_urinary, 3, 0)

    # Microorganism
    if microorganism_col in df.columns:
        is_kleb = df[microorganism_col].astype(str) \
            .str.lower().str.contains('kleb|other', na=False)
        score += np.where(is_kleb, 2, 0)

    # Inappropriate Abx
    if inapprop_abx_col in df.columns: score += np.where(df[inapprop_abx_col] == 1, 2, 0)

    return score


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

    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "PALACION-BAENA (2019)"

    # 1. Validation: Check for missing columns before calculating
    validate_required_columns(df,
        required_cols=[age_col, charlson_col, pitt_col, sirs_col,
            bsi_not_urinary_col, is_non_ecoli_col, inapprop_abx_col],
        score_name=score_name
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Age > 50', 'col': age_col, 'condition': df[age_col] > 50, 'points': 3},
        {'desc': 'Charlson > 3', 'col': charlson_col, 'condition': df[charlson_col] > 3, 'points': 4},
        {'desc': 'Pitt Score >= 6', 'col': pitt_col, 'condition': df[pitt_col] >= 6, 'points': 3},
        {'desc': 'SIRS >= 2', 'col': sirs_col, 'condition': df[sirs_col] >= 2, 'points': 4},
        {'desc': 'Non-Urinary Source', 'col': bsi_not_urinary_col, 'condition': df[bsi_not_urinary_col] == 1, 'points': 3},
        {'desc': 'Non-E. coli', 'col': is_non_ecoli_col, 'condition': df[is_non_ecoli_col] == 1, 'points': 2},
        {'desc': 'Inappropriate Abx', 'col': inapprop_abx_col, 'condition': df[inapprop_abx_col] == 1, 'points': 2}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)


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
    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "HOLMGREN 2020"

    validate_required_columns(df,
        required_cols=[hosp_abroad_col, prev_culture_col, prev_swab_col],
        score_name=score_name
    )

    rules = [
        {'desc': 'Hosp abroad (12m)', 'col': hosp_abroad_col, 'condition': df[hosp_abroad_col] == 1, 'points': 1},
        {'desc': 'Prev 3GCR Culture', 'col': prev_culture_col, 'condition': df[prev_culture_col] == 1, 'points': 1},
        {'desc': 'Prev 3GCR Swab', 'col': prev_swab_col, 'condition': df[prev_swab_col] == 1, 'points': 1}
    ]

    return evaluate_score(df, rules, score_name, verbose, logger)


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
    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "GAVAGHAN 2025"

    validate_required_columns(df,
        required_cols=[age_col, prior_esbl_col, nursing_home_col,
            urinary_catheter_col, prior_abx_col],
        score_name=score_name
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Prior ESBL (365 days)', 'col': prior_esbl_col, 'condition': df[prior_esbl_col] == 1, 'points': 4},
        {'desc': 'Age >= 65', 'col': age_col, 'condition': df[age_col] >= 65, 'points': 1},
        {'desc': 'Nursing Home Resident', 'col': nursing_home_col, 'condition': df[nursing_home_col] == 1, 'points': 2},
        {'desc': 'Urinary Catheter Present', 'col': urinary_catheter_col, 'condition': df[urinary_catheter_col] == 1,'points': 1},
        {'desc': 'Prior Antibiotics (90d)', 'col': prior_abx_col, 'condition': df[prior_abx_col] == 1, 'points': 2}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)


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
    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "JONES 2025"

    validate_required_columns(df,
        required_cols=[prior_esbl_col, prior_abx_col,
            chronic_dialysis_col, transfer_hosp_col],
        score_name=score_name
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Prior ESBL (180 days)', 'col': prior_esbl_col, 'condition': df[prior_esbl_col] == 1, 'points': 5},
        {'desc': 'Prior Antibiotics (30 days)', 'col': prior_abx_col, 'condition': df[prior_abx_col] == 1, 'points': 2},
        {'desc': 'Chronic Dialysis', 'col': chronic_dialysis_col, 'condition': df[chronic_dialysis_col] == 1, 'points': 2},
        {'desc': 'Transfer from Hospital', 'col': transfer_hosp_col, 'condition': df[transfer_hosp_col] == 1, 'points': 1}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)


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
    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "TUMBARELLO 2019"

    validate_required_columns(df,
        required_cols=[prior_esbl_col, hosp_90d_col,
            abx_90d_col, urinary_catheter_col],
        score_name=score_name
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Prior ESBL History', 'col': prior_esbl_col, 'condition': df[prior_esbl_col] == 1, 'points': 4},
        {'desc': 'Recent Hospitalization (90d)', 'col': hosp_90d_col, 'condition': df[hosp_90d_col] == 1, 'points': 2},
        {'desc': 'Recent Antibiotics (90d)', 'col': abx_90d_col, 'condition': df[abx_90d_col] == 1, 'points': 2},
        {'desc': 'Urinary Catheter', 'col': urinary_catheter_col, 'condition': df[urinary_catheter_col] == 1, 'points': 1}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)


def calculate_kim_score(df,
                        prior_esbl_col='hx_prior_esbl_any',
                        hosp_1y_col='hx_hosp_last_365d',
                        nursing_home_col='hx_nursing_home_resident',
                        urinary_catheter_col='hx_urinary_catheter_present',
                        prior_abx_90d_col='hx_prior_abx_90d',
                        **kwargs):
    """
    Computes the Kim et al. (2019) ESBL Risk Score.

    !!! warning "Binary Inputs Required"
        This function strictly expects **binary flags (1 or 0)** for all parameters.

    ??? note "Clinical Criteria & Point Allocation (Click to expand)"
        It focuses on identifying risk factors specifically for community-onset BSIs caused
        by ESBL-producing E. coli and Klebsiella species[cite: 6]. This model is particularly useful
        for differentiating resistant from susceptible strains right at the point of admission[cite: 6].

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
    # Safely extract logging options from kwargs
    verbose = kwargs.get('verbose', False)
    logger = kwargs.get('logger', None)
    score_name = "KIM 2019"

    validate_required_columns(df,
        required_cols=[prior_esbl_col, hosp_1y_col,
             nursing_home_col, urinary_catheter_col, prior_abx_90d_col],
        score_name=score_name
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Prior ESBL History', 'col': prior_esbl_col, 'condition': df[prior_esbl_col] == 1, 'points': 5},
        {'desc': 'Recent Hospitalization (1y)', 'col': hosp_1y_col, 'condition': df[hosp_1y_col] == 1, 'points': 2},
        {'desc': 'Nursing Home Resident', 'col': nursing_home_col, 'condition': df[nursing_home_col] == 1, 'points': 2},
        {'desc': 'Urinary Catheter', 'col': urinary_catheter_col, 'condition': df[urinary_catheter_col] == 1, 'points': 1},
        {'desc': 'Prior Antibiotic Use (90d)', 'col': prior_abx_90d_col, 'condition': df[prior_abx_90d_col] == 1, 'points': 1}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, score_name, verbose, logger)


def calculate_consensus_2023_meta(df,
                                  prior_esbl_col='prior_esbl_history',
                                  prior_abx_col='prior_abx_90d',
                                  hosp_col='recent_hospitalization',
                                  invasive_proc_col='recent_procedure',
                                  verbose=False,
                                  logger=None):
    """
    Weighted Consensus Score based on the Timbrook & Fowler (2023) Meta-Analysis.

    Weights are derived from the most common aORs reported in the review.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe.
    prior_esbl_col : str, default='prior_esbl_history'
        Column name for prior ESBL history.
    prior_abx_col : str, default='prior_abx_90d'
        Column name for prior antibiotic exposure.
    hosp_col : str, default='recent_hospitalization'
        Column name for recent hospitalization.
    invasive_proc_col : str, default='recent_procedure'
        Column name for recent invasive procedure.
    verbose : bool, default=False
        Whether to print verbose logs.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    pd.Series
        A pandas Series containing the computed score for each patient.

    References
    ----------
    Timbrook & Fowler (2023) Meta-Analysis.
    """
    # Added validation for consistency with the other scoring functions
    validate_required_columns(df,
        required_cols=[prior_esbl_col, prior_abx_col, hosp_col, invasive_proc_col],
        score_name="Consensus 2023 Meta"
    )

    # Pure Logic Definition
    rules = [
        {'desc': 'Prior ESBL',                'col': prior_esbl_col,    'condition': df[prior_esbl_col] == 1,    'points': 4},
        {'desc': 'Recent Hospitalization',    'col': hosp_col,          'condition': df[hosp_col] == 1,          'points': 2},
        {'desc': 'Antibiotic Exposure',       'col': prior_abx_col,     'condition': df[prior_abx_col] == 1,     'points': 2},
        {'desc': 'Recent Invasive Procedure', 'col': invasive_proc_col, 'condition': df[invasive_proc_col] == 1, 'points': 1}
    ]

    # Execution & Debugging
    return evaluate_score(df, rules, "Consensus 2023 Meta", verbose, logger)