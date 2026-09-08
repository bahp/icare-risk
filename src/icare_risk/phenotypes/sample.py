# icare_risk/phenotypes/vitals.py
import pandas as pd
from icare_risk.core.dataset import ClinicalDataset

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
    return results.reset_index().rename(columns={'is_fever': 'fever_phenotype'})