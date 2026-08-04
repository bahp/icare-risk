"""
Author:
Date:

This module acts as the "translation layer" between raw, messy Electronic
Health Record (EHR) data and clean, standardized clinical variables.

NAMING CONVENTIONS
==================
To keep the pipeline predictable and easy to read, all functions in this file
follow strict naming prefixes based on the exact type of data they return:

1. `has_` (Returns Integer 1 or 0)
   Used for past medical history, chronic conditions, or comorbidities.
   Example: `has_diabetes()`, `has_congestive_heart_failure()`

2. `is_` (Returns Integer 1 or 0)
   Used for acute, current patient states occurring during this specific encounter.
   Example: `is_mechanically_ventilated()`, `is_urinary_source()`

3. `derive_` (Returns Continuous, Categorical, or Multi-level Data)
   Used when the output is not a simple boolean flag. This includes extracting
   numbers, categorizing statuses, or performing date math.
   Example: `derive_age_at_admission()`, `derive_fever_status()`git
"""

# src/phenotypes.py
import pandas as pd
import numpy as np

# Silence the downcasting warning globally
pd.set_option('future.no_silent_downcasting', True)

from icare_risk.utils import (check_col_bool,
                              check_col_contains,
                              check_col_threshold,
                              check_col_icd10)

# ------------------------------------------------------------------------
#                                  HELPER METHODS
# ------------------------------------------------------------------------
def _patient_has_historical_codes(df, context_df,
                                      patient_col,
                                      code_col,
                                      target_codes):
    """
    Internal helper to identify patients who possess specific clinical codes
    in their history.

    This function performs a vectorized search across a contextual data source
    (e.g., a 'problems' or 'diagnoses' table) to find matches for a list of
    target medical codes (ICD-10, Read, or SNOMED). It supports hierarchical
    matching by using prefix string matching (e.g., a search for 'I21' will
    correctly catch 'I21.9').

    The resulting boolean mask is automatically aligned with the index or
    'patient_id' column of the primary features DataFrame, ensuring it can be
    assigned directly as a new feature column.

    Notes
    -----
    - The function normalizes all codes to lowercase and strips whitespace for robust matching.
    - Performance: Uses vectorized pandas string operations; however, the loop over
      `target_codes` may be slow if the list of targets is extremely large.

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame (typically pivoted time-series).
    context_df (pd.DataFrame):
        The contextual table to search (e.g., self.context_dfs['problems']).
    patient_col (str):
        The column name in the context_df that identifies the patient.
    code_col (str):
        The column name in the context_df containing the medical codes.
    target_codes (list of str):
        The list of medical code prefixes to search for.

    Returns
    -------
        pd.Series (bool): A boolean series indexed identically to `df`, where True
            indicates the patient has at least one matching historical code.
    """
    if context_df is None or context_df.empty or code_col not in context_df.columns:
        return pd.Series(False, index=df.index)

    # Normalize targets for safe string matching
    targets = [str(c).lower().strip() for c in target_codes]

    # Drop empty rows to speed up the string search
    valid_rows = context_df.dropna(subset=[code_col]).copy()
    series_lower = valid_rows[code_col].astype(str).str.lower()

    # Build a mask for any row that starts with any of our target codes
    # (Using startswith allows 'I21' to catch 'I21.9')
    match_mask = pd.Series(False, index=valid_rows.index)
    for t in targets:
        match_mask = match_mask | series_lower.str.startswith(t)

    # Extract the unique patient IDs who had a matching code
    patients_with_condition = valid_rows.loc[match_mask, patient_col].unique()

    # Align with the main time-series dataframe
    # Check if patient_id is in the index (it usually is after our pivot setup)
    if 'patient_id' in df.index.names:
        current_patients = df.index.get_level_values('patient_id')
    else:
        current_patients = df['patient_id']

    return current_patients.isin(patients_with_condition)


def has_medication_in_window(df, **kwargs):
    """
    Evaluates if a specific medication (or class of medications) was administered
    to the patient within a defined clinical time window.

    This function acts as a high-level phenotype extractor, typically used to
    identify the administration of vasopressors (for Pitt Score) or specific
    empirical antibiotics (for INCREMENT-ESBL or 3GCR resistance logic). It relies
    on the 'pharmacy' context to perform a fuzzy string match against prescribed
    drug names.

    The "window" is usually calculated relative to the start of the clinical
    encounter or the collection time of a positive blood culture.

    Notes
    -----
    - The function performs case-insensitive substring matching.
    - It utilizes the internal `_get_prescriptions_in_window` helper to handle
      the complex date-math and multi-table joining between the clinical
      timeline and the pharmacy records.
    - Any missing data or patients without pharmacy records default to 0.

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame (the recipient of the flag).
    **kwargs: Arbitrary keyword arguments containing:
        target_meds (list of str): The names or partial names of drugs to search
            for (e.g., ['norepinephrine', 'levophed']).
        window_hours (int, optional): The duration of the look-forward or
            look-back period in hours. Defaults to 24.
        context_dfs (dict): Dictionary containing the 'pharmacy' DataFrame.

    Returns
    -------
        np.ndarray: An array of integers (1 or 0) aligned with `df.index`, where 1
            indicates that at least one medication from `target_meds` was found
            within the specified `window_hours`.
    """
    target_meds = kwargs.get('target_meds', [])
    window_hours = kwargs.get('window_hours', 24)
    context = kwargs.get('context_dfs', {})

    # Use your existing helper to get the list of meds for each row
    prescribed_series = _get_prescriptions_in_window(df, context, window_hours=window_hours)

    if prescribed_series is None:
        return pd.Series(0, index=df.index).values

    # Normalize the target list to lowercase
    target_meds_lower = [str(m).lower() for m in target_meds]

    def check_meds(med_list):
        if not isinstance(med_list, list) or len(med_list) == 0:
            return 0
        # Flatten the patient's meds into a single string for easy searching
        meds_str = " ".join([str(m).lower() for m in med_list])
        # Return 1 if ANY of our target meds are found in the patient's string
        return 1 if any(tm in meds_str for tm in target_meds_lower) else 0

    result = prescribed_series.apply(check_meds)

    # Realign with the main index and fill missing with 0
    return result.reindex(df.index).fillna(0).values

def _get_nearest_micro_record(df,
                              context_dfs,
                              cols_to_fetch,
                              tolerance_days=3):
    """
    Internal helper to perform a temporal "nearest-match" join between the
    primary clinical timeline and microbiology results.

    In clinical datasets, microbiology results (e.g., blood cultures) often have
    different collection timestamps than vital signs. This function uses
    `pd.merge_asof` to synchronize the two, ensuring that for any given clinical
    observation, we are looking at the most relevant (closest in time)
    microbiology data.

    Notes:
        - The join is performed using a "nearest" direction, meaning it will pick
          the culture closest in time, whether it was collected slightly before
          or after the clinical observation.
        - Requires 'SUBJECT' in microbiology and 'patient_id' in the main DataFrame
          to serve as the join keys.

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame (must have 'patient_id' and 'date' in
        the index or columns).
    context_dfs (dict):
        Dictionary containing the 'microbiology' DataFrame.
    cols_to_fetch (list):
        Specific columns to retrieve from the microbiology records (e.g., 'ORGANISM_BUG', 'SITE').
    tolerance_days (int, optional):
        The maximum allowable time difference between the clinical date and
        the culture collection date. Defaults to 3.

    Returns
    -------
    pd.DataFrame: A DataFrame indexed by `[patient_id, date]` containing the
        requested microbiology columns, or None if the microbiology context
        is missing.
    """
    df_micro = context_dfs.get('microbiology')
    if df_micro is None or df_micro.empty:
        return None

    # 1. Prepare Main DF (Reset index to expose 'patient_id' and 'date')
    df_temp = df.reset_index()
    df_temp['date'] = pd.to_datetime(df_temp['date'])

    # 2. Prepare Micro DF
    micro_temp = df_micro.copy()
    micro_temp['LATEST_COLLECT_DT'] = pd.to_datetime(micro_temp['LATEST_COLLECT_DT'])
    micro_temp = micro_temp.sort_values('LATEST_COLLECT_DT')

    # 3. Temporal Join
    merged = pd.merge_asof(
        df_temp.sort_values('date'),
        micro_temp[['SUBJECT', 'LATEST_COLLECT_DT'] + cols_to_fetch],
        left_on='date',
        right_on='LATEST_COLLECT_DT',
        left_by='patient_id',
        right_by='SUBJECT',
        direction='nearest',
        tolerance=pd.Timedelta(days=tolerance_days)
    )

    # Return indexed by the original MultiIndex for easy re-alignment
    return merged.set_index(['patient_id', 'date'])


def _get_prescriptions_in_window(df, context_dfs, window_hours=24):
    """
    Internal helper to identify all medications prescribed within a specific
    temporal window relative to a clinical event.

    This function is primarily used to identify "Empirical Therapy"—medications
    given shortly after a suspected infection (represented by the 'date' in
    the features DataFrame) but before definitive lab results are available.
    It filters the 'pharmacy' context to find prescriptions that fall within
    [0, window_hours] of the record date.

    Notes:
        - The join is performed on both 'patient_id' and 'ENCNTR_ID' to ensure
          medications are linked to the correct hospital admission.
        - The function calculates the difference in hours between `ORDER_DT_TM`
          and the record `date`.
        - Results are grouped by patient and date to allow for multi-drug
          regimen analysis (e.g., checking for combination therapy).

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame.
    context_dfs (dict):
        Dictionary containing the 'pharmacy' DataFrame.
    window_hours (int, optional):
        The length of the observation window in hours. Defaults to 24.

    Returns
    -------
    pd.Series: A Series indexed by `[patient_id, date]` where each value is
        a Python list containing the names of medications (`MEDICATION_NAME_SHORT`)
        prescribed within that window. Returns None if pharmacy data is missing.
    """
    df_pharm = context_dfs.get('pharmacy')
    if df_pharm is None or df_pharm.empty:
        return None

    # Align dates and reset index
    df_temp = df.reset_index()
    df_temp['date'] = pd.to_datetime(df_temp['date'])

    pharm_temp = df_pharm.copy()
    pharm_temp['ORDER_DT_TM'] = pd.to_datetime(pharm_temp['ORDER_DT_TM'])

    # Join on patient and encounter
    merged = pd.merge(
        df_temp,
        pharm_temp[['SUBJECT', 'ENCNTR_ID', 'ORDER_DT_TM', 'MEDICATION_NAME_SHORT']],
        left_on=['patient_id', 'ENCNTR_ID'],
        right_on=['SUBJECT', 'ENCNTR_ID'],
        how='left'
    )

    # Filter for medications given within the first X hours of the record 'date'
    # In INCREMENT-ESBL, 'appropriate' usually refers to the first 24h of BSI
    merged['diff_hours'] = (merged['ORDER_DT_TM'] - merged['date']).dt.total_seconds() / 3600
    valid_meds = merged[(merged['diff_hours'] >= 0) & (merged['diff_hours'] <= window_hours)]

    # Group back so each time-series row has a list of meds
    return valid_meds.groupby(['patient_id', 'date'])['MEDICATION_NAME_SHORT'].apply(list)



def derive_historical_condition(df, **kwargs):
    """
    Identifies if a patient has a specific condition in their medical
    history (1=Yes, 0=No).

    This function acts as a centralized extractor for chronic comorbidities by
    searching multiple iCARE contextual sources. It prioritizes the 'problems'
    list (active problem lists) and the 'diagnoses' table (historical discharge
    summaries/secondary diagnoses). It is the primary engine used to populate
    the 17 binary flags required for the Charlson Comorbidity Index (CCI).

    Clinical Logic:
    - Returns 1 if any medical code in the patient's history starts with a code
      provided in `target_codes`.
    - Merges results from both the current Problem List and historical Episode Diagnoses.

    iCARE Mapping:
    - Source 1: `icare_problems_anon` (Columns: `SUBJECT`, `PROBLEM_CODE`)
    - Source 2: `icare_episodes_diagnosis_anon` (Columns: `SUBJECT`, `DIAGNOSIS_CODE`)

    Notes:
        - The function relies on the internal `_patient_has_historical_codes` helper
          to perform the string-prefix matching.
        - If `target_codes` is empty or the required contexts are missing, the
          function defaults to 0 for all patients.

    Parameters
    ----------
    df (pd.DataFrame): The primary features DataFrame (pivoted time-series).
    **kwargs: Arbitrary keyword arguments containing:
        context_dfs (dict): Dictionary of supporting DataFrames (must contain
            'problems' and optionally 'diagnoses').
        target_codes (list of str): The ICD-10 or Read-code prefixes to
            identify the condition (e.g., ['I21', 'I22'] for MI).

    Returns
    -------
    np.ndarray: An array of integers (1 or 0) aligned with `df.index`.
    """
    flag = pd.Series(0, index=df.index)
    context = kwargs.get('context_dfs', {})
    target_codes = kwargs.get('target_codes', [])

    if not target_codes:
        return flag.values

    # 1. Search the Problems Context
    has_prob = _patient_has_historical_codes(
        df=df,
        context_df=context.get('problems'),
        patient_col='SUBJECT',
        code_col='PROBLEM_CODE',
        target_codes=target_codes
    )
    flag.loc[has_prob] = 1

    # 2. Search the Episode Diagnoses Context
    has_diag = _patient_has_historical_codes(
        df=df,
        context_df=context.get('diagnoses'),
        patient_col='SUBJECT',
        # Update this column name if your iCARE diagnosis table uses something else (e.g., ICD_CODE)
        code_col='DIAGNOSIS_CODE',
        target_codes=target_codes
    )
    flag.loc[has_diag] = 1

    return flag.values




def map_medical_codes(df, code_col, target_codes, **kwargs):
    """
    Generic function to identify if any code in a patient's record
    matches a list of target diagnostic codes (ICD-10, ICD-9, etc.).

    Parameters:
    -----------
    df : pd.DataFrame
    code_col : str
        The column containing patient codes (can be a list or a string).
    target_codes : list
        The list of codes that define the clinical category (e.g., ['I50', 'I11.0']).
    """

    def check_match(patient_codes):
        if pd.isna(patient_codes):
            return 0
        # Convert to list if it's a string (e.g., "I10, I50")
        if isinstance(patient_codes, str):
            patient_codes = [c.strip() for c in patient_codes.split(',')]

        # Check if there is any intersection between patient codes and targets
        # This handles partial matches too (e.g., 'E11' matches 'E11.9')
        match = any(any(str(pc).startswith(str(tc)) for tc in target_codes)
                    for pc in patient_codes)
        return 1 if match else 0

    return df[code_col].apply(check_match)


# ---------------------------------------
def derive_diabetes_status(df, insulin_col='med_insulin_given', glucose_col='Glucose_24h_max'):
    """
    Derives a diabetes diagnosis (1=Yes, 0=No) based on clinical proxies
    rather than a direct medical history column.

    Rules for positive phenotype:
    1. Patient was administered insulin.
    2. OR, Patient had a severely elevated rolling glucose reading (> 200).
    """
    # Start by assuming the patient does NOT have diabetes (fill with 0)
    status = pd.Series(0, index=df.index)

    # Rule A: Check for the medication proxy
    if insulin_col in df.columns:
        # If insulin is 1, set status to 1. Otherwise, keep the current status.
        status = np.where(df[insulin_col] == 1, 1, status)

    # Rule B: Check for the lab value proxy
    if glucose_col in df.columns:
        # If glucose is > 200, set status to 1. Otherwise, keep current status.
        status = np.where(df[glucose_col] > 200.0, 1, status)

    return status


def derive_liver_disease(df, ast_col='AST', alt_col='ALT', cirrhosis_med_col='med_lactulose'):
    """
    Example 2: Deriving Liver Disease based on severely elevated liver enzymes
    or the administration of Lactulose (a common medication for hepatic encephalopathy).
    """
    status = pd.Series(0, index=df.index)

    # Rule A: Medication proxy
    if cirrhosis_med_col in df.columns:
        status = np.where(df[cirrhosis_med_col] == 1, 1, status)

    # Rule B: Lab proxy (Enzymes > 3x the upper limit of normal)
    if ast_col in df.columns and alt_col in df.columns:
        status = np.where((df[ast_col] > 120) | (df[alt_col] > 150), 1, status)

    return status



# -----------------------------------------------------------------------------------
#                 CHARLSON COMORBIDITY INDEX (CCI) EXTRACTORS
# -----------------------------------------------------------------------------------
# These methods extract the 17 chronic conditions required to compute the Charlson
# Comorbidity Index, a validated method of categorizing comorbidities of patients
# based on the International Classification of Diseases (ICD) diagnosis codes.
#
# Reference:
# Charlson ME, et al. "A new method of classifying prognostic comorbidity in
# longitudinal studies: development and validation." J Chronic Dis. 1987;40(5):373-83.
def has_myocardial_infarction(df, **kwargs):
    """""
    Identifies history of MI.
    Codes: Use 'MI' category from CCI Gold CSV (e.g., 323..00, G30..00).
    Search: `icare_problem_anon` and `icare_episodes_diagnosis_anon`.
    pass
    """
    pass

def has_congestive_heart_failure(df, **kwargs):
    """
    @Example:

    Determines if a patient has a history of Congestive Heart Failure (CHF).

    Clinical Logic:
    A patient is flagged with CHF if ANY of the following criteria are met:
    1. ICD-10 Codes: Patient record contains 'I50' (Heart failure) or
       'I11.0' (Hypertensive heart disease with heart failure).
    2. Explicit Flag: The `hx_chf` boolean column is exactly 1.
    3. Medication Proxy: The patient is actively prescribed specific heart failure
       medications like 'entresto', 'milrinone', or 'dobutamine' in the `home_meds` column.

    Required Columns in df:
    - `diagnosis_codes` (String/List of codes)
    - `hx_chf` (Int/Float 1.0 or 0.0)
    - `home_meds` (String)

    Returns:
    pd.Series of integers (1 for has CHF, 0 for no CHF).
    """
    pass

def has_peripheral_vascular_disease(df, **kwargs):
    """"""
    pass

def has_cerebrovascular_disease(df, **kwargs):
    """"""
    pass

def has_dementia(df, **kwargs):
    """"""
    pass

def has_chronic_pulmonary_disease(df, **kwargs):
    """"""
    pass

def has_connective_tissue_disease(df, **kwargs):
    """"""
    pass

def has_peptic_ulcer_disease(df, **kwargs):
    """"""
    pass

def has_mild_liver_disease(df, **kwargs):
    """

    Clinical Logic:
    1. Explicit History: (Fill this out)
    2. ICD-10 Codes: (Fill this out - e.g., K70, K74)
    3. Medications: (Fill this out - e.g., lactulose)
    4. Lab Values: (Fill this out - e.g., AST > 3x normal, Bilirubin > 2.0)
    """

    # c1_history = check_col_bool(...)
    # c2_icd10 = check_col_icd10(...)
    # c3_meds = check_col_contains(...)
    # c4_labs = check_col_threshold(...)

    # combined_signal = c1_history | c2_icd10 | c3_meds | c4_labs
    # return combined_signal.astype(int)
    pass


def has_diabetes(df, config=None):
    """
    @Example:

    Determines if a patient has diabetes using a multi-modal data approach.

    Clinical Logic defined by the student:
    We consider a patient to have diabetes if ANY of the following are true:
    1. Explicit History: The `diabetes_history` boolean column is exactly 1.
    2. ICD-10 Codes: The `diagnosis_codes` column contains E10, E11, or E14.
    3. Medications: The `medications_given` column contains 'insulin' or 'metformin'.
    4. Lab Values: The rolling maximum glucose (`glucose_max_24h`) is > 200 mg/dL.

    Returns:
    pd.Series of integers (1 for has_diabetes, 0 for no diabetes).
    """
    # 1. Extract signals using helpers
    c1_history = check_col_bool(df, 'diabetes_history')
    c2_icd10 = check_col_icd10(df, 'diagnosis_codes', target_codes=['E10', 'E11', 'E14'])
    c3_meds = check_col_contains(df, 'medications_given', keywords=['insulin', 'metformin'])
    c4_labs = check_col_threshold(df, 'glucose_max_24h', threshold=200, operator='>')

    # 2. Combine signals (Logical OR: if any signal is True, the condition is met)
    combined_signal = c1_history | c2_icd10 | c3_meds | c4_labs

    # 3. Return as 1s and 0s
    return combined_signal.astype(int)

def has_diabetes_without_complications(df, **kwargs):
    """"""
    pass

def has_diabetes_with_complications(df, **kwargs):
    """"""
    pass

def has_hemiplegia_or_paraplegia(df, **kwargs):
    """"""
    pass

def has_moderate_to_severe_renal_disease(df, **kwargs):
    """"""
    pass

def has_malignancy(df, **kwargs):
    """"""
    pass

def has_moderate_to_severe_liver_disease(df, **kwargs):
    """"""
    pass

def has_metastatic_solid_tumor(df, **kwargs):
    """"""
    pass

def has_aids(df, **kwargs):
    """"""
    pass


# ----------------------------------------------------------------------------------------
#                          PITT BACTEREMIA SCORE EXTRACTORS
# ----------------------------------------------------------------------------------------
# These methods derive the 5 acute severity variables required to compute the Pitt
# Bacteremia Score, a validated measure of acute illness severity in patients with
# bloodstream infections (BSIs) assessed either at the day of positive culture or
# admission.
#
# Reference:
# Korvick JA, et al. "Prospective observational study of Klebsiella bacteremia..."
# Clin Infect Dis. 1992;15(5):795-801.
# (Also widely validated for ESBL by Paterson et al., 2004).

def derive_mental_status_score(df, **kwargs):
    """
    Maps neurological status to Pitt Bacteremia Score weights.

    Clinical Logic:
    - Alertness: 0 pts
    - Disorientation: 1 pt
    - Stupor: 2 pts
    - Coma: 4 pts

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Check Columns: `news_conscious_level_score`,
                     `new_score_consciousness`, or
                     `glasgow_coma_score`.
    """
    pass

def derive_pitt_fever_status(df, **kwargs):
    """
    Clinical Logic:
    - 36.1°C – 38.9°C: 0 pts
    - 35.1°C – 36.0°C or 39.0°C – 39.9°C: 1 pt
    - <= 35°C or >= 40°C: 2 pts

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (10933766, 486347689)
    - Value: `observation_result_clean`

    Returns
    -------
    pd.Series (int)
        Discrete points (0, 1, or 2) for the Pitt Score.
    """
    temp_cols = kwargs.get('temp_cols', ['temp'])
    if isinstance(temp_cols, str):
        temp_cols = [temp_cols]

    # Initialize master score at 0 points
    master_score = pd.Series(0, index=df.index)

    for col in temp_cols:
        if col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce')

            # 1. Initialize a temporary Pandas Series for this column
            current_score = pd.Series(0, index=df.index)
            # 2. Logic: 1 pt for 35.1-36.0 OR 39.0-39.9
            mask_1pt = ((temp >= 35.1) & (temp <= 36.0)) | ((temp >= 39.0) & (temp <= 39.9))
            current_score.loc[mask_1pt] = 1
            # 3. Logic: 2 pts for <= 35.0 OR >= 40.0
            mask_2pt = (temp <= 35.0) | (temp >= 40.0)
            current_score.loc[mask_2pt] = 2
            # 4. Retain the highest score observed across the provided columns
            master_score = master_score.combine(current_score, max)

    return master_score


def derive_pitt_hypotension_status(df, **kwargs):
    """
    Derives the hypotension component of the Pitt Bacteremia Score.
    Returns 2 points if SBP < 90 OR if the vasopressor flag is 1.

    Clinical Logic:
    - Presence of hypotension: +2 pts
    - Systolic BP < 90 mmHg or requires vasopressors from icare_antitbiotic_prescribing whiuch contains vast range of drugs.

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (13389125)
    - Value: `observation_result_clean`

    """
    flag = pd.Series(0, index=df.index)

    # 1. Check Vitals (SBP < 90)
    sbp_col = kwargs.get('sbp_col', 'sbp_24h_min')
    if sbp_col in df.columns:
        sbp = pd.to_numeric(df[sbp_col], errors='coerce')
        flag.loc[sbp < 90.0] = 1

    # 2. Check Vasopressors
    vp_col = kwargs.get('vasopressor_col', 'has_vasopressors')
    if vp_col in df.columns:
        vp_flag = pd.to_numeric(df[vp_col], errors='coerce')
        flag.loc[vp_flag == 1] = 1

    return flag

def is_mechanically_ventilated(df, **kwargs):
    """Checks respiratory support flowsheets.

    Clinical Logic:
    - Receiving mechanical ventilation: +2 pts
    - Search respiratory support flowsheets.

    Returns
    -------
    pd.Series (int)
        1 if ventilated, 0 otherwise.
    """
    pass

def has_recent_cardiac_arrest(df, **kwargs):
    """Checks resuscitation codes/events.

     Clinical Logic:
    - Cardiac arrest within window: +4 pts
    - Check resuscitation event codes.
    """
    pass


# ----------------------------------------------------------------------------------------
#                               SIRS CRITERIA EXTRACTORS                               ---
# ----------------------------------------------------------------------------------------
# These methods evaluate the 4 physiological parameters required to determine if a
# patient meets the criteria for Systemic Inflammatory Response Syndrome (SIRS),
# indicating a severe, systemic immune response to infection.

# Reference:
# Bone RC, et al. "Definitions for sepsis and organ failure and guidelines for the
# use of innovative therapies in sepsis." Chest. 1992;101(6):1644-55.
def derive_sirs_tachycardia(df, **kwargs):
    """
    Identifies if a patient meets the heart rate criteria for SIRS.

    Clinical Logic:
    - Returns 1 if Heart Rate (HR) > 90 beats per minute (bpm).
    - This threshold represents the cardiovascular component of the SIRS
      criteria, signaling a systemic stress response or early sepsis.

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` == 13472364
    - Value: `observation_result_clean`

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame containing pivoted vital sign data.
    **kwargs:
        Arbitrary keyword arguments containing:
        hr_col (str, optional): The column name for the heart rate.
            Defaults to 'hr_24h_max' to capture the most severe
            tachycardic event in the clinical window.

    Returns
    -------
    np.ndarray: An array of integers (1 or 0) aligned with `df.index`,
        where 1 indicates the patient meets the SIRS tachycardia criteria.
        Identifies Tachycardia for SIRS criteria (1=Yes, 0=No).
    """
    flag = pd.Series(0, index=df.index)

    # Get the heart rate column (defaulting to a 24h max if using rolling windows)
    hr_col = kwargs.get('hr_col', 'hr_24h_max')

    if hr_col in df.columns:
        hr = pd.to_numeric(df[hr_col], errors='coerce')
        flag.loc[hr > 90.0] = 1

    return flag


def derive_sirs_tachypnea(df, **kwargs):
    """
    Identifies if a patient meets the respiratory criteria for SIRS.

    Clinical Logic:
    - Returns 1 if Respiratory Rate (RR) > 20 breaths/min OR
    - Partial pressure of arterial carbon dioxide (PaCO2) < 32 mmHg.
    - This reflects either direct tachypnea or compensatory respiratory 
      alkalosis due to hyperventilation.

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` == 9096705
    - Value: `observation_result_clean`

    Paramters
    ---------
    df (pd.DataFrame): 
        Primary features DataFrame containing pivoted respiratory or arterial blood gas data.
    **kwargs:
        rr_col (str, optional): 
            Column name for the respiratory rate. Defaults to 'rr_24h_max'.
        paco2_col (str, optional): 
            Column name for the PaCO2 lab result. Defaults to 'paco2'.

    Returns
    -------
    np.ndarray: Binary array (1/0) aligned with `df.index`.
    """
    flag = pd.Series(0, index=df.index)

    rr_col = kwargs.get('rr_col', 'rr_24h_max')
    paco2_col = kwargs.get('paco2_col', 'paco2')  # Optional, if available in labs

    if rr_col in df.columns:
        rr = pd.to_numeric(df[rr_col], errors='coerce')
        flag.loc[rr > 20.0] = 1

    # Layer on the PaCO2 check if the lab data is present
    if paco2_col in df.columns:
        paco2 = pd.to_numeric(df[paco2_col], errors='coerce')
        flag.loc[paco2 < 32.0] = 1

    return flag


def derive_sirs_abnormal_temp(df, **kwargs):
    """"
    Identifies if a patient meets the temperature criteria for SIRS.

    Clinical Logic:
    - Returns 1 if Temperature > 38.0°C (Fever) or < 36.0°C (Hypothermia).
    - Checks multiple columns (e.g., max/min rolling windows) to capture
      extremes within a 24-hour period.

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (10933766, 486347689)
    - Value: `observation_result_clean`

    Parameters
    ----------
    df (pd.DataFrame):
        Primary features DataFrame with pivoted vitals.
    **kwargs:
        temp_cols (list/str): Column names to evaluate (e.g., 'temp_24h_max').

    Returns
    -------
    np.ndarray: Binary array (1/0) aligned with `df.index`.
    """
    flag = pd.Series(0, index=df.index)

    # Accept a list to check both max and min rolling windows for extremes
    temp_cols = kwargs.get('temp_cols', ['temp_24h_max', 'temp_24h_min', 'temp'])
    if isinstance(temp_cols, str):
        temp_cols = [temp_cols]

    for col in temp_cols:
        if col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce')
            flag.loc[(temp > 38.0) | (temp < 36.0)] = 1

    return flag





def derive_sirs_abnormal_wbc(df, **kwargs):
    """
    Identifies if a patient meets the leukocyte (WBC) criteria for SIRS.

    The SIRS criteria define an abnormal White Blood Cell count as a significant
    leukocytosis, leukopenia, or a shift to immature forms (bandemia). This
    function evaluates the primary WBC count and, if available, the percentage
    of band neutrophils.

    Clinical Logic (SIRS 1992 Consensus):
        - WBC count > 12.0 x 10^9/L (Leukocytosis) OR
        - WBC count < 4.0 x 10^9/L (Leukopenia) OR
        - Immature neutrophils (Bands) > 10%

    iCARE Mapping:
        - Table: `icare_pathology_blood_anon`
        - Filter: `test_code` contains 'wbc'
        - Value: `result_cleaned`

    Notes:
        - The function assumes the WBC unit is 10^9/L (e.g., a value of 12.5
          represents 12,500 cells/µL). If the raw data uses absolute counts,
          the thresholds (12.0 and 4.0) must be adjusted to 12000 and 4000.
        - `pd.to_numeric` with `errors='coerce'` is used to handle non-numeric
          artifacts in the lab results.

    Parameters
    ----------
    df (pd.DataFrame):
        The primary features DataFrame containing pivoted laboratory values.
    **kwargs: Arbitrary keyword arguments containing:
        wbc_col (str, optional):
            The column name for the WBC count. Defaults to 'wbc'.
        bands_col (str, optional):
            The column name for the immature band percentage. Defaults to 'bands'.

    Returns
    -------
    np.ndarray: An array of integers (1 or 0) aligned with `df.index`,
        where 1 indicates the patient meets the SIRS WBC criteria.
    """
    flag = pd.Series(0, index=df.index)

    wbc_col = kwargs.get('wbc_col', 'wbc')
    bands_col = kwargs.get('bands_col', 'bands')  # Optional, if available

    if wbc_col in df.columns:
        wbc = pd.to_numeric(df[wbc_col], errors='coerce')
        # NOTE: This assumes your WBC is stored in thousands (e.g., 12.0). 
        # If your raw iCARE data stores absolute values (e.g., 12000), 
        # change the thresholds below to 12000.0 and 4000.0.
        flag.loc[(wbc > 12.0) | (wbc < 4.0)] = 1

    if bands_col in df.columns:
        bands = pd.to_numeric(df[bands_col], errors='coerce')
        flag.loc[bands > 10.0] = 1

    return flag

# ----------------------------------------------------------------------------------------
#                        INCREMENT-ESBL DIRECT CLINICAL EXTRACTORS
# ----------------------------------------------------------------------------------------
# These methods extract the specific standalone clinical variables (source of infection,
# microorganism type, and antibiotic appropriateness) required to compute the final
# INCREMENT-ESBL risk score for 30-day mortality.

# Reference:
# Palacios-Baena Z, et al. "Development and validation of the INCREMENT-ESBL predictive
# score for mortality in patients with bloodstream infections..." J Antimicrob Chemother.
# 2017;72(3):906-913.
def derive_age_at_admission(df, **kwargs):
    """Already available.

    iCARE Mapping:
    - Table: `icare_episodes_anon`
    - Column: `age_at_admission`
    """
    pass


def derive_bsi_not_urinary(df, **kwargs):
    """
    Determines if the source of the Bloodstream Infection (BSI) is NON-urinary.
    In the INCREMENT-ESBL score, non-urinary sources (respiratory, abdominal, etc.)
    are associated with higher mortality and receive +3 points.

    Clinical Logic:
    A patient is flagged as having a NON-urinary source (returns 1) UNLESS we can
    prove the source was urinary. We assume urinary if:
    1. Explicit Flag: The `bsi_source` column explicitly equals 'Urinary'.
    2. Concurrent Cultures: A urine culture (`urine_culture_result`) drawn within 48h
       grew the same organism.
    3. ICD-10 Proxy: The patient was billed for a UTI (e.g., 'N39.0') on admission.

    If none of the urinary criteria are met, we default to NON-urinary (1).

    Required Columns in df:
    - `bsi_source` (String)
    - `urine_culture_result` (String)
    - `diagnosis_codes` (String/List)

    iCARE mapping:
    iCARE Mapping:
    - Table: `icare_episodes_diagnosis_anon`
    - Column: `diagnosis_code_snomed`
    - Logic: use the diagnosis icd-10 codes
    - J15.8, J15.9, J18.0, J18.1, J18.9, J85.1, N39.0, N10,
      N13.6, N30.0, N30.9 (to cover sputum and urine, for example).

    Returns
    -------
    pd.Series (int)
        1 if source is non-urinary, 0 if urinary.
    """
    urine_code = kwargs.get('urine_culture_code', 'LOINC-630-4')
    merged = _get_nearest_micro_record(df, kwargs.get('context_dfs', {}), ['ORDER_CODE'])

    if merged is None:
        return pd.Series(1, index=df.index).values

    # Logic: 1 if NOT the urine code
    is_not_urinary = (merged['ORDER_CODE'] != urine_code).astype(int)
    return is_not_urinary.reindex(df.index).fillna(1).values


def derive_is_non_ecoli(df, **kwargs):
    """
    Identifies if the ESBL-producing organism is a non-E. coli species
    (e.g., Klebsiella, Enterobacter, Serratia, Proteus).

    Clinical Logic:
    E. coli bacteremia generally has a better prognosis in this specific cohort.
    We flag the patient as 'Non-E. coli' if:
    1. Blood Culture Result: The `microorganism` or `blood_culture_org` column
       contains keywords like 'klebsiella', 'enterobacter', 'serratia', or 'other'.
    2. Exclusion: Ensure we do NOT flag it if the string strictly says
       'escherichia coli' or 'e. coli'.

    Required Columns in df:
    - `microorganism` or `blood_culture_org` (String from microbiology LIS system)

    Returns
    -------
    pd.Series (int)
        1 if non-E. coli Enterobacteriaceae, 0 if E. coli or No Growth.
    """
    merged = _get_nearest_micro_record(df,
        kwargs.get('context_dfs', {}), ['ORGANISM_BUG'])

    if merged is None:
        return pd.Series(0, index=df.index).values

    # Logic: 1 if contains bacteria that isn't E. coli
    bug_str = merged['ORGANISM_BUG'].astype(str).str.lower()
    is_ecoli = bug_str.str.contains('e. coli|escherichia', na=False)
    is_growth = ~bug_str.str.contains('no growth|nan', na=False)

    is_non_ecoli = (is_growth & ~is_ecoli).astype(int)
    return is_non_ecoli.reindex(df.index).fillna(0).values


def derive_abx_inappropriate(df, **kwargs):
    """
    Determines if the empirical antibiotic therapy administered was INAPPROPRIATE.

    Clinical Logic:
    Empirical therapy is the drug given *before* the final lab results come back.
    Therapy is considered INAPPROPRIATE (returns 1) if:
    1. Resistance: The `empiric_abx_given` (medication given in the first 24h)
       matches a drug listed in the `abx_resistant_to` column (from the lab report).
    2. No Coverage: The patient received no active anti-ESBL antibiotics
       (like carbapenems) within the first 24 hours of blood culture collection.
    3. Explicit Flag: If an `inappropriate_abx_flag` already exists from a
       clinical pharmacist's manual review, use it.

    Required Columns in df:
    - `empiric_abx_given` (String/List of medications)
    - `abx_resistant_to` (String/List from susceptibility report)
    - `inappropriate_abx_flag` (Int 1 or 0, optional)

    Note:
    This method  requires us to look at Pharmacy data (empiric_abx_given) and
    Laboratory data (abx_resistant_to) at the same time.

    iCARE Mapping:
    - Merge `icare_microbiology_anon` with `icare_pharmacy_prescribing_anon`.
    - Compare `sample_collected_dt` (`icare_pathology_blood_anon`)
         with `order_dt_tm` (`icare_pharmacy_prescribing_anon`).
    - Inappropriate if administration > 1 day from blood cultures or
      > 4 days without targeted anti-ESBL therapy (e.g. Carbapenems)

    Implementation
    --------------

    Returns 1 if NO 'Susceptible' antibiotic was given within 24h of the culture.

    Evaluates if empirical antibiotic therapy was inappropriate.

    Compares 'microbiology' sensitivity results against 'pharmacy'
    prescriptions. Inappropriate therapy (+2 points in INCREMENT-ESBL)
    occurs if the organism is resistant to all administered drugs.

    Returns
    -------
    pd.Series (int)
        1 if therapy was inappropriate, 0 otherwise.
    """

    context = kwargs.get('context_dfs', {})

    # 1. Get the list of meds given in the first 24h
    prescribed_series = _get_prescriptions_in_window(df, context)

    # 2. Get the micro sensitivity record
    micro_record = _get_nearest_micro_record(df, context, ['SENSITIVITY', 'ORGANISM_BUG'])

    if prescribed_series is None or micro_record is None:
        return pd.Series(1, index=df.index).values  # Default to inappropriate (risk)

    # Combine for comparison
    comparison_df = micro_record.copy()
    comparison_df['prescribed_meds'] = prescribed_series

    def check_logic(row):
        # If the bug is Resistant or the drug wasn't in the 'Susceptible' list
        # In this synthetic version, we simplify:
        # If Micro says 'Resistant', we flag it as inappropriate (1)
        if row['SENSITIVITY'] == 'Resistant':
            return 1
        # If they didn't receive any meds at all
        if not isinstance(row['prescribed_meds'], list) or len(row['prescribed_meds']) == 0:
            return 1
        return 0

    inappropriate = comparison_df.apply(check_logic, axis=1)
    return inappropriate.reindex(df.index).fillna(1).values


def determine_bsi_source(df, **kwargs):
    """Implemented above in <derive_bsi_not_urinary>."""
    pass

def identify_microorganism_type(df, **kwargs):
    """Implemented above <derive_is_non_ecoli>."""
    pass

def evaluate_antibiotic_appropriateness(df, **kwargs):
    """Implemented above <derive_abx_inappropriate>."""
    pass


def derive_ground_truth(df, **kwargs):
    """
    Assigns a random binary ground truth label (1 or 0) to each patient.

    This ensures that all rows belonging to the same patient have the
    exact same label, which is required for stay-level classification
    (e.g., predicting if a patient will develop ESBL during this stay).

    Parameters
    ----------
    df (pd.DataFrame): The primary features DataFrame.
    **kwargs:
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns
    -------
        np.ndarray: A binary array (1/0) aligned with df.index.
    """
    seed = kwargs.get('seed', 42)

    # 1. Identify unique patients in the current batch
    # We use the index level 'patient_id' if available, otherwise a column
    if 'patient_id' in df.index.names:
        unique_patients = df.index.get_level_values('patient_id').unique()
    else:
        unique_patients = df['patient_id'].unique()

    # 2. Generate one random label for each unique patient
    np.random.seed(seed)
    random_labels = np.random.randint(0, 2, size=len(unique_patients))

    # 3. Map the labels back to the full dataframe
    mapping = dict(zip(unique_patients, random_labels))

    if 'patient_id' in df.index.names:
        return df.index.get_level_values('patient_id').map(mapping).values
    else:
        return df['patient_id'].map(mapping).values