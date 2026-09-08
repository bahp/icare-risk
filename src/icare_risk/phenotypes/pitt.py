import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Helper Phenotypes
# -----------------------------------------------------------------------------
def is_mechanically_ventilated(dataset):
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

def has_recent_cardiac_arrest(dataset):
    """Checks resuscitation codes/events.

     Clinical Logic:
    - Cardiac arrest within window: +4 pts
    - Check resuscitation event codes.

    Returns
    -------
    pd.Series (int)
        1 if cardiac arrest, 0 otherwise.
    """
    pass


def has_vasopressors(dataset, window_hours=24, target_meds=[]):
    """

    Parameters
    ----------
    dataset:
    window_hours:
    target_meds:

    Returns
    -------
    """
    pass


# -----------------------------------------------------------------------------
# Helper Pitt Partial Scores
# -----------------------------------------------------------------------------
def compute_pitt_fever_score(dataset):
    """Derives the fever status component of the Pitt Bacteremia Score
    
    iCARE Mapping
    -------------
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (10933766, 486347689)
    - Value: `observation_result_clean`

    Clinical Logic
    --------------
    - 36.1°C – 38.9°C: 0 pts
    - 35.1°C – 36.0°C or 39.0°C – 39.9°C: 1 pt
    - <= 35°C or >= 40°C: 2 pts

    Parameters
    ----------
    """
    import numpy as np
    vitals = dataset.get_current_stay('vitals')
    print(vitals.OBSERVATION_NAME.unique())
    mask = vitals['OBSERVATION_NAME'] == 'Body Temperature'
    temp_vitals = vitals[mask].copy()
    
    temp = pd.to_numeric(temp_vitals['OBSERVATION_RESULT_CLEAN'], errors='coerce')
    
    # Assign points based on established clinical ranges
    conditions = [
        (temp <= 35.0) | (temp >= 40.0),   # Extreme: 2 points
        (temp >= 35.1) & (temp <= 36.0),   # Mildly low: 1 point
        (temp >= 39.0) & (temp <= 39.9)    # Mildly high: 1 point
    ]
    temp_vitals['fever_score'] = np.select(conditions, [2, 1, 1], default=0)
    
    # Take the maximum points scored during this specific admission
    results = temp_vitals.groupby(['SUBJECT', 'std_admission_time'])['fever_score'].max()
    return results.reset_index(name='pitt_fever_score')


def compute_pitt_mental_score(dataset):
    vitals = dataset.get_current_stay('vitals')
    mask = vitals['OBSERVATION_NAME'] == 'Glasgow Coma Scale'
    gcs_vitals = vitals[mask].copy()
    
    gcs = pd.to_numeric(gcs_vitals['OBSERVATION_RESULT_CLEAN'], errors='coerce')
    
    # Map GCS to Pitt mental status points
    conditions = [
        (gcs <= 9),                    # Comatose: 4 points
        (gcs >= 10) & (gcs <= 12),     # Stuporous: 2 points
        (gcs >= 13) & (gcs <= 14)      # Disoriented: 1 point
    ]
    gcs_vitals['mental_score'] = np.select(conditions, [4, 2, 1], default=0)
    
    results = gcs_vitals.groupby(['SUBJECT', 'std_admission_time'])['mental_score'].max()
    return results.reset_index()


def compute_pitt_hypo_score(dataset):
    """Derives hypotension component of the Pitt Bacteremia Score.
    
    Returns 2 points if SBP < 90 OR if the vasopressor flag is 1.

    Clinical Logic:
    - Presence of hypotension: +2 pts
    - Systolic BP < 90 mmHg or requires vasopressors from icare_antitbiotic_prescribing whiuch contains vast range of drugs.

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (13389125)
    - Value: `observation_result_clean`
    """
    vitals = dataset.get_current_stay('vitals')
    mask = vitals['OBSERVATION_NAME'] == 'Systolic Blood Pressure'
    sbp_vitals = vitals[mask].copy()
    
    sbp = pd.to_numeric(sbp_vitals['OBSERVATION_RESULT_CLEAN'], errors='coerce')
    # SBP < 90 is the standard proxy for acute hypotension
    sbp_vitals['hypo_score'] = np.where(sbp < 90.0, 2, 0)
    
    results = sbp_vitals.groupby(['SUBJECT', 'std_admission_time'])['hypo_score'].max()
    return results.reset_index(name='pitt_hypo_score')


def compute_pitt_vent_score(dataset):
    # Depending on your schema, this might be in vitals or interventions
    interventions = dataset.get_current_stay('interventions')
    mask = interventions['INTERVENTION_NAME'] == 'Mechanical Ventilation'
    vent_data = interventions[mask].copy()
    
    # If the row exists, the patient was ventilated
    vent_data['vent_score'] = 2 
    
    results = vent_data.groupby(['SUBJECT', 'std_admission_time'])['vent_score'].max()
    return results.reset_index()


def compute_pitt_arrest_score(dataset):
    interventions = dataset.get_current_stay('interventions')
    mask = interventions['INTERVENTION_NAME'] == 'Cardiac Arrest'
    arrest_data = interventions[mask].copy()
    
    # If the row exists, the patient had an arrest
    arrest_data['arrest_score'] = 4
    
    results = arrest_data.groupby(['SUBJECT', 'std_admission_time'])['arrest_score'].max()
    return results.reset_index()

import functools

def compute_pitt_score(dataset):
    # 1. Execute independent component calculations
    fever = compute_pitt_fever_score(dataset)
    mental = compute_pitt_mental_score(dataset)
    hypo = compute_pitt_hypo_score(dataset)
    vent = compute_pitt_vent_score(dataset)
    arrest = compute_pitt_arrest_score(dataset)
    
    # 2. Merge all dataframes on the standardized admission keys
    dataframes = [fever, mental, hypo, vent, arrest]
    merged = functools.reduce(
        lambda left, right: pd.merge(left, right, on=['SUBJECT', 'std_admission_time'], how='outer'), 
        dataframes
    )
    
    # 3. Treat missing data (e.g., patient had no GCS measured) as a normal baseline of 0 points
    score_cols = ['fever_score', 'mental_score', 'hypo_score', 'vent_score', 'arrest_score']
    merged[score_cols] = merged[score_cols].fillna(0)
    
    # 4. Calculate total Pitt score
    merged['pitt_total_score'] = merged[score_cols].sum(axis=1)
    
    return merged