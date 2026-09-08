import numpy as np
import pandas as pd
from icare_risk.phenotypes.utils import DEFAULT_SCHEMA


def flag_if_any(dataset, rules, output_name, schema=DEFAULT_SCHEMA):
    """
    Evaluates clinical rules across multiple tables.
    `rules` format: ('table', ['Names'], ['Codes'], lambda logic)
    """
    processed_tables = []
    unique_tables = set(rule[0] for rule in rules)
    
    for table_name in unique_tables:
        df = dataset.get_current_stay(table_name)
        if df.empty:
            continue
            
        # 1. Look up the schema for this table
        name_col = schema[table_name].get('name_col')
        code_col = schema[table_name].get('code_col')
        value_col = schema[table_name].get('value_col')
            
        df['calc_flag'] = 0
        val = pd.to_numeric(df[value_col], errors='coerce')
        
        # 2. Process rules specific to this table
        table_rules = [r for r in rules if r[0] == table_name]
        
        for _, target_names, target_codes, logic_func in table_rules:
            target_names = target_names or []
            target_codes = target_codes or []
            
            # Build the mask dynamically based on what was provided
            mask = pd.Series(False, index=df.index)
            
            if target_names and name_col in df.columns:
                mask = mask | df[name_col].isin(target_names)
                
            if target_codes and code_col in df.columns:
                mask = mask | df[code_col].isin(target_codes)
                
            # Apply the logic flag
            df.loc[mask & logic_func(val), 'calc_flag'] = 1
            
        processed_tables.append(df[['SUBJECT', 'std_admission_time', 'calc_flag']])
        
    if not processed_tables:
        return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', output_name])
        
    combined = pd.concat(processed_tables, ignore_index=True)
    results = combined.groupby(['SUBJECT', 'std_admission_time'])['calc_flag'].max()
    
    return results.reset_index(name=output_name)


# ----------------------------------------------------------------------------------------
#                               SIRS CRITERIA EXTRACTORS                               ---
# ----------------------------------------------------------------------------------------
# These methods evaluate the 4 physiological parameters required to determine if a
# patient meets the criteria for Systemic Inflammatory Response Syndrome (SIRS),
# indicating a severe, systemic immune response to infection.

# Reference:
# Bone RC, et al. "Definitions for sepsis and organ failure and guidelines for the
# use of innovative therapies in sepsis." Chest. 1992;101(6):1644-55.
def derive_sirs_tachycardia(dataset):
    """
    Identifies if a patient meets the heart rate criteria for SIRS.

    iCARE Mapping
    -------------
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` == 13472364 (or name == 'Heart Rate')
    - Value: `observation_result_clean`

    Clinical Logic
    --------------
    - Returns 1 if Heart Rate (HR) > 90 beats per minute (bpm).
    - This threshold represents the cardiovascular component of the SIRS
      criteria, signaling a systemic stress response or early sepsis.
    """
    return flag_if_any(
        dataset=dataset,
        output_name='sirs_tachycardia_flag',
        rules=[
            ('vitals',     ['Heart Rate'],    [13472364],    lambda val: val > 90.0)
        ]
    )


def derive_sirs_tachypnea(dataset):
    return flag_if_any(
        dataset=dataset,
        output_name='sirs_tachypnea_flag',
        rules=[
            # ('Table',    ['Names'],               ['Codes'],     Logic)
            ('vitals',     ['Respiratory Rate'],    [9096705],     lambda val: val > 20.0),
            ('pathology',  ['PaCO2', 'pCO2'],       [],            lambda val: val < 32.0),
        ]
    )


def derive_sirs_abnormal_temp(dataset):
    """Identifies if a patient meets the temperature criteria for SIRS.

    Clinical Logic:
    - Returns 1 if Temperature > 38.0°C (Fever) or < 36.0°C (Hypothermia).

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
    return flag_if_any(
        dataset=dataset,
        output_name='sirs_abnormal_temp_flag',
        rules=[
            (
                'vitals',                                  # 1. Table
                ['Body Temperature'],                      # 2. Names
                [10933766, 486347689],                     # 3. Codes
                lambda val: (val > 38.0) | (val < 36.0)    # 4. Logic
            )
        ]
    )


def derive_sirs_abnormal_wbc(dataset, **kwargs):
    """
    Identifies if a patient meets the leukocyte (WBC) criteria for SIRS.

    !!! warning "Unit Assumption"
        This function assumes the WBC unit is **10^9/L** (e.g., a value of 12.5 represents
        12,500 cells/µL). If the raw data uses absolute counts or a different unit, the
        thresholds must be manually adjusted in the code (e.g. to 12000 and 4000).

    The SIRS criteria define an abnormal White Blood Cell count as a significant
    leukocytosis, leukopenia, or a shift to immature forms (bandemia). This
    function evaluates the primary WBC count and, if available, the percentage
    of band neutrophils.

    ??? note "Clinical Logic & iCARE Mapping (Click to expand)"
        **Clinical Logic (SIRS 1992 Consensus):**

        * **Leukocytosis:** WBC count > 12.0 x 10^9/L
        * **Leukopenia:** WBC count < 4.0 x 10^9/L
        * **Bandemia:** Immature neutrophils (Bands) > 10%

        **iCARE Mapping:**

        * **Table:** `icare_pathology_blood_anon`
        * **Filter:** `test_code` contains 'wbc'
        * **Value:** `result_cleaned`

        **Implementation Notes:**

        * `pd.to_numeric` with `errors='coerce'` is used to handle non-numeric
          artifacts safely in the lab results.

    Parameters
    ----------
    df : pandas.DataFrame
        The primary features DataFrame containing pivoted laboratory values.
    **kwargs
        Arbitrary keyword arguments containing:
        * `wbc_col` (str): Column name for the WBC count. Defaults to 'wbc'.
        * `bands_col` (str): Column name for the immature band percentage. Defaults to 'bands'.
        * `wbc_high` (float): The upper threshold for leukocytosis. Defaults to 12.0.
        * `wbc_low` (float): The lower threshold for leukopenia. Defaults to 4.0.
        * `bands_threshold` (float): The threshold for bandemia. Defaults to 10.0.

    Returns
    -------
    numpy.ndarray
        An array of integers (1 or 0) aligned with `df.index`, where 1 indicates
        the patient meets the SIRS WBC criteria.
    """
    # 1. Get dynamic thresholds from kwargs (with safe clinical defaults)
    wbc_high = kwargs.get('wbc_high', 12.0)
    wbc_low = kwargs.get('wbc_low', 4.0)
    bands_threshold = kwargs.get('bands_threshold', 10.0)

    # 2. Let the helper function handle the extraction and grouping
    return flag_if_any(
        dataset=dataset,
        output_name='sirs_abnormal_wbc_flag',
        rules=[
            # Rule 1: Leukocytosis or Leukopenia
            (
                'pathology',                                        # Table
                ['White Blood Cells', 'WBC', 'Leukocytes'],         # Names
                ['wbc'],                                            # Codes
                lambda val: (val > wbc_high) | (val < wbc_low)      # Logic
            ),
            # Rule 2: Bandemia (Immature Neutrophils)
            (
                'pathology',
                ['Bands', 'Immature Granulocytes', 'Band Neutrophils'],
                ['bands'], 
                lambda val: val > bands_threshold
            )
        ]
    )

