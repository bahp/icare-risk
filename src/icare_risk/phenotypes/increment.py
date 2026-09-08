# icare_risk/phenotypes/increment.py
import pandas as pd
from icare_risk.phenotypes.sirs import flag_if_any
from icare_risk.phenotypes.utils import DEFAULT_SCHEMA

def derive_bsi_not_urinary(dataset, output_name: str, **kwargs):
    """Returns 1 (Non-urinary source) UNLESS a Urine Culture is present."""
    urine_codes = kwargs.get('urine_culture_codes', ['LOINC-630-4'])

    # Find ANY evidence of a Urine Culture. lambda val: True means we don't care 
    # what the bug is, just the fact that the culture was ordered counts.
    urinary_evidence = flag_if_any(
        dataset=dataset,
        output_name='is_urinary_temp',
        rules=[
            ('microbiology', [], urine_codes, lambda val: True)
        ]
    )
    
    # Merge with cohort to handle patients with NO microbiology data
    cohort = dataset.get_cohort()
    merged = pd.merge(cohort, urinary_evidence, on=['SUBJECT', 'std_admission_time'], how='left')
    
    # Fill missing with 0 (No urine culture found)
    merged['is_urinary_temp'] = merged['is_urinary_temp'].fillna(0).astype(int)
    
    # Invert the logic (1 - 0 = 1 Non-Urinary, 1 - 1 = 0 Urinary)
    merged[output_name] = 1 - merged['is_urinary_temp']
    return merged[['SUBJECT', 'std_admission_time', output_name]]


def derive_bsi_non_ecoli(dataset, output_name: str, **kwargs):
    """
    Returns 1 if Blood Culture grew an organism that is NOT E. coli.
    Ignores 'E. coli' and 'No Growth'.
    """
    blood_culture_codes = kwargs.get('blood_culture_codes', ['LOINC-600-7'])
    
    df = dataset.get_current_stay('microbiology')
    cohort = dataset.get_cohort()
    
    if df.empty:
        cohort[output_name] = 0
        return cohort
        
    # Filter only for Blood Cultures
    bc = df[df['ORDER_CODE'].isin(blood_culture_codes)].copy()
    bc['is_non_ecoli'] = 0
    
    # Clean the strings for safe matching
    orgs = bc['ORGANISM_BUG'].astype(str).str.lower()
    
    # The Exclusion Logic
    is_ecoli = orgs.str.contains('e. coli|escherichia', regex=True)
    is_negative = orgs.str.contains('no growth|not applicable', regex=True)
    
    # Flag as 1 if it is NEITHER E. coli NOR Negative
    bc.loc[(~is_ecoli) & (~is_negative), 'is_non_ecoli'] = 1
    
    # Group and merge
    results = bc.groupby(['SUBJECT', 'std_admission_time'])['is_non_ecoli'].max().reset_index(name=output_name)
    merged = pd.merge(cohort, results, on=['SUBJECT', 'std_admission_time'], how='left')
    merged[output_name] = merged[output_name].fillna(0).astype(int)
    
    return merged[['SUBJECT', 'std_admission_time', output_name]]