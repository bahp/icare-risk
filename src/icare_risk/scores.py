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
    """
    Computes the Modified Early Warning Score (MEWS).
    Standardized to use the evaluate_score engine for auditing.

    .. TODO: Revisit the logic, it has to be wrong, not covering the
             different ranges properly.
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
    """
    Computes the Age-Adjusted Charlson Comorbidity Index (CCI).
    Standardized to use the evaluate_score engine for auditing.
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
    """
    Computes the Charlson Comorbidity Index using Quan et al. weights
    and hierarchy rules. Safely handles missing columns and dynamic names.
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
    """
    Computes the Pitt Bacteremia Score (0 to 14 points).

    Expects pre-computed points for temperature and mental status (0, 1, 2, 4),
    and boolean 1/0 flags for hypotension, ventilation, and cardiac arrest.
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
    Simply adds up the pre-computed 1/0 flags.
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

    Reference: Palacios-Baena et al., J Antimicrob Chemother, 2017.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                      | Condition Evaluated               | Points |
    |----------------------------------------|-----------------------------------|--------|
    | Demographics                           | Age > 50 years                    |   +3   |
    | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
    | Acute Underlying Severity              | High Pitt bacteremia score        |   +3   |
    | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
    | SIRS Severity                          | Severe SIRS / Shock present       |   +4   |
    | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
    | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='age'
        Column name for patient age (numeric).
    severe_comorb_col : str, default='severe_comorbidities'
        Column name for severe comorbidities flag (binary 1/0).
    high_pitt_col : str, default='high_pitt_score'
        Column name for high Pitt bacteremia score flag (binary 1/0).
    bsi_source_col : str, default='bsi_source'
        Column name for infection source (string).
    severe_sirs_col : str, default='severe_sirs_or_shock'
        Column name for severe SIRS/shock flag (binary 1/0).
    microorganism_col : str, default='microorganism'
        Column name for isolated organism (string).
    inapprop_abx_col : str, default='inappropriate_abx'
        Column name for inappropriate antibiotic therapy flag (binary 1/0).

    Returns
    -------
    pd.Series
        A pandas Series containing the computed INCREMENT-ESBL score (integers)
        for each patient, matching the input DataFrame index.
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
    """
    Computes the INCREMENT-ESBL predictive score for mortality.
    Reference: Palacios-Baena et al., J Antimicrob Chemother, 2017.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                      | Condition Evaluated               | Points |
    |----------------------------------------|-----------------------------------|--------|
    | Demographics                           | Age > 50 years                    |   +3   |
    | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
    | Acute Underlying Severity              | High Pitt bacteremia score (>= 6) |   +3   |
    | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
    | SIRS Severity                          | Severe SIRS / Shock (SIRS >= 2)   |   +4   |
    | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
    | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

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
    Reference: Holmgren et al., "An easy-to-use scoring system...", 2020.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                    | Condition Evaluated                    | Points |
    |--------------------------------------|----------------------------------------|--------|
    | Hospital care abroad                 | Hospitalized abroad in last 12 months  |   +1   |
    | Previous 3GCR Culture                | Previous 3GCR in blood or urine        |   +1   |
    | Previous 3GCR Rectal Swab            | Previous 3GCR in rectal swab           |   +1   |

    Interpretation: Score >= 1 is considered "High Risk" in low-resistance settings.
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

# src/scores.py

def calculate_gavaghan_score(df,
                             age_col='AGE_AT_ADMISSION',
                             prior_esbl_col='hx_prior_esbl_365d',
                             nursing_home_col='hx_nursing_home_resident',
                             urinary_catheter_col='hx_urinary_catheter_present',
                             prior_abx_col='hx_prior_fc_abx_90d',
                             **kwargs):
    """
    Computes the Gavaghan et al. (2025) ESBL Risk Score.
    Reference: Gavaghan et al., Antimicrob Steward Healthc Epidemiol, 2025.

    The Gavaghan et al. (2025) score is a contemporary tool developed specifically for
    risk assessment of ESBL-producing Enterobacterales bacteremia in a tertiary setting.

    Clinical Variable & Point Allocation:
    | Clinical Variable                    | Condition Evaluated                       | Points |
    |--------------------------------------|-------------------------------------------|--------|
    | Prior ESBL                           | Any ESBL organism within 365 days         |   +4   |
    | Age                                  | Age >= 65 years                           |   +1   |
    | Nursing Home Resident                | Lives in a long-term care facility        |   +2   |
    | Urinary Catheter                     | Indwelling catheter at presentation      |   +1   |
    | Prior Antibiotics                    | Fluoroquinolone or Cephalosporin (90d)    |   +2   |
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


# src/scores.py

def calculate_jones_score(df,
                          prior_esbl_col='hx_prior_esbl_180d',
                          prior_abx_col='hx_prior_abx_30d',
                          chronic_dialysis_col='hx_chronic_dialysis',
                          transfer_hosp_col='hx_transfer_from_hosp',
                          **kwargs):
    """
    Computes the Jones et al. (2025) ESBL Risk Score for Non-Urinary Isolates.
    Reference: Jones et al., Pharmacotherapy, 2025.

    The Jones et al. (2025) score is a specialized tool designed specifically for
    non-urinary isolates. This is a crucial distinction in clinical practice, as
    risk factors for ESBL in bloodstream or respiratory infections often differ
    from those in simple UTIs.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Positive ESBL culture within 180 days     |   +5   |
    | Prior Antibiotics           | Any antibiotic use within 30 days         |   +2   |
    | Chronic Dialysis            | Patient on hemodialysis or peritoneal     |   +2   |
    | Transfer from Hospital      | Admission via transfer from another hosp  |   +1   |
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
    Reference: Int J Antimicrob Agents, 2019 (Utrecht/Stockholm Cohort).

    This specific model is designed for community-onset sepsis, making it a vital
    baseline for patients arriving at the Emergency Department before hospital-acquired
    factors come into play.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Known colonization/infection (any time)   |   +4   |
    | Recent Hospitalization      | Hospitalized within last 90 days          |   +2   |
    | Recent Antibiotics          | Beta-lactams/Quinolones within 90 days    |   +2   |
    | Urinary Catheter            | Permanent or recent urinary catheter      |   +1   |

    Interpretation: High risk is typically defined as a score >= 3 or 4.
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

# src/scores.py

def calculate_kim_score(df,
                        prior_esbl_col='hx_prior_esbl_any',
                        hosp_1y_col='hx_hosp_last_365d',
                        nursing_home_col='hx_nursing_home_resident',
                        urinary_catheter_col='hx_urinary_catheter_present',
                        prior_abx_90d_col='hx_prior_abx_90d',
                        **kwargs):
    """
    Computes the Kim et al. (2019) ESBL Risk Score.
    Reference: Kim et al., J Korean Med Sci, 2019.

    It focuses on identifying risk factors specifically for community-onset BSIs caused
    by ESBL-producing E. coli and Klebsiella species. This model is particularly useful
    for differentiating resistant from susceptible strains right at the point of admission.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Prior ESBL colonization or infection      |   +5   |
    | Recent Hospitalization      | Hospitalization within the last 1 year    |   +2   |
    | Nursing Home Resident       | Resident in a long-term care facility     |   +2   |
    | Urinary Catheter            | Use of indwelling urinary catheter        |   +1   |
    | Prior Antibiotics           | Use of antibiotics within 90 days         |   +1   |
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