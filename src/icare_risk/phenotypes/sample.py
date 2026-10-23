# icare_risk/phenotypes/vitals.py
import pandas as pd
from icare_risk.core.dataset import ClinicalDataset
from icare_risk.core.decorators import clinical_feature

@clinical_feature(
    output_name='is_fever',
    tables={'vitals': ['OBSERVATION_NAME', 'OBSERVATION_RESULT_CLEAN']},
    agg_func='any'
)
def compute_fever_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    # 1. Filter to Temperature rows
    temp_mask = df['OBSERVATION_NAME'].str.contains('Temperature', case=False, na=False)
    temp_vitals = df[temp_mask].copy()

    # 2. Apply clinical logic
    temp_vitals['numeric_result'] = pd.to_numeric(temp_vitals['OBSERVATION_RESULT_CLEAN'], errors='coerce')
    temp_vitals['is_fever'] = temp_vitals['numeric_result'] > 37.5

    return temp_vitals

def compute_fever(dataset: ClinicalDataset) -> pd.DataFrame:
    # 1. Safely fetch ONLY vitals during the current admission
    vitals = dataset.get_current_stay('vitals')
    
    # 2. Guard clause in case the query returns no data
    if vitals.empty:
        return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', 'fever_phenotype'])
        
    # 3. Filter to Temperature rows and EXPLICITLY CREATE A COPY
    # (Adjust 'Temperature' to match exactly how it appears in your OBSERVATION_NAME)
    temp_mask = vitals['OBSERVATION_NAME'].str.contains('Temperature', case=False, na=False)
    temp_vitals = vitals[temp_mask].copy()
    
    if temp_vitals.empty:
         return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', 'fever_phenotype'])
    
    # 4. Apply clinical logic
    # It's good practice to force the result column to numeric in case of string artifacts
    temp_vitals['numeric_result'] = pd.to_numeric(temp_vitals['OBSERVATION_RESULT_CLEAN'], errors='coerce')
    temp_vitals['is_fever'] = temp_vitals['numeric_result'] > 37.5
    
    # 5. Group by the standardized episode identifiers
    results = temp_vitals.groupby(['SUBJECT', 'std_admission_time'])['is_fever'].any()
    
    # 6. Reset index and give the final column a clear clinical name
    return results.reset_index()

def compute_hypoxia(dataset: ClinicalDataset):
    """[Example] Computes hypoxia metrics for a given clinical dataset.

    This is a placeholder function provided as an example.

    Parameters
    ----------
    dataset : ClinicalDataset
        The initialized data provider engine.

    Returns
    -------
    None
    """
    # Safely fetches ONLY vitals during the current admission
    vitals = dataset.get_current_stay('vitals')

    # Filter to Oxygen Saturation rows AND explicitly create a copy
    o2_mask = vitals['OBSERVATION_NAME'] == 'Oxygen Saturation'
    o2_vitals = vitals[o2_mask].copy()

    # Logic: is Oxygen < 90% at any point during this stay?
    # This will now execute perfectly without warnings
    o2_vitals['is_hypoxic'] = o2_vitals['OBSERVATION_RESULT_CLEAN'] < 90.0

    # Group by the standardized columns
    results = o2_vitals.groupby(['SUBJECT', 'std_admission_time'])['is_hypoxic'].any()

    return results.reset_index()

def compute_test_phenotype(dataset: ClinicalDataset):
    vitals = dataset.get_current_stay('vitals')
    pathology = dataset.get_current_stay('pathology')
    problems = dataset.get_current_stay('problems')

    print(vitals.shape, vitals.columns.tolist())
    print(pathology.shape, pathology.columns.tolist())
    print(vitals.SUBJECT.nunique())

    hx_vitals = dataset.get_historical('vitals')
    v2 = dataset.get_current_stay('vitals', time_window_hours=2)
    print(vitals.shape)
    print(hx_vitals.shape)
    print(v2.shape)

    return v2