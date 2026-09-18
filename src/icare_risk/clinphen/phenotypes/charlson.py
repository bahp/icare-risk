import pandas as pd

from icare_risk.clinphen.utils.filtering import match_codes
from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype


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

@phenotype(
    name="has_diabetes",
    domains=["problems", "prescribing"],
    category="charlson_components"
)
#def has_diabetes(df, config=None):
def has_diabetes(ctx: EpisodeContext,
                 codes: list,
                 window: tuple = ("0h", "24h"),
                 **kwargs) -> int:
    """Determines if a patient has diabetes using a multi-modal data approach.

    Notes
    -----
    Implemented with multimodal.

    Clinical Logic
    --------------
    We consider a patient to have diabetes if ANY of the following are true:
    1. Explicit History: The `diabetes` code appears in 'problems'.
    2. Explicit Diagnosis: The `diabetes' code appears in 'diagnosis` -> Not implemented.
    3. Medication Proxy: The `prescribing` table contains ['insulin', 'metformin', 'gliclazide'].
    4. Lab Values Proxy: The rolling maximum glucose (`glucose_max_24h`) is > 200 mg/dL.

    Returns
    -------
    pd.Series of integers (1 for has_diabetes, 0 for no diabetes).
    """
    pass

def has_congestive_heart_failure(df, **kwargs):
    """Determines if a patient has a history of Congestive Heart Failure (CHF).

    Notes
    -----
    Implemented with multimodal.

    Clinical Logic
    --------------
    We consider a patient to have CHF if ANY of the following are true:
    1. Explicit History: The `heart failure' code appears in 'problems'.
    2. Explicit Diagnosis: The `heart failure' code appears in 'diagnosis` -> Not implemented.
    3. Medication Proxy: The patient is actively prescribed specific heart failure
       medications like 'entresto', 'milrinone', or 'dobutamine' in 'prescribing' current
       episode or stay.

    Returns:
    pd.Series of integers (1 for has CHF, 0 for no CHF).
    """
    pass

def has_mild_liver_disease(df, **kwargs):
    """Determines if a patient has mild liver disease.

    Notes
    -----
    Implemented with multimodal.

    Clinical Logic
    --------------
    1. Explicit History: The `mild liver disease' code appears in 'problems'.
    2. ICD-10 Codes: (Fill this out - e.g., K70, K74)
    3. Medication Proxy: The `prescribing` table contains ['lactulose'].
    4. Lab Values Proxy: The 'pathology' table contains (Bilirubin > 2.0)
    """
    pass

def has_peripheral_vascular_disease(df, **kwargs):
    """Identifies if a patient has a history of peripheral vascular disease."""
    pass

def has_cerebrovascular_disease(df, **kwargs):
    """Identifies if a patient has a history of cerebrovascular disease."""
    pass

def has_dementia(df, **kwargs):
    """Identifies if a patient has a documented history of dementia."""
    pass

def has_chronic_pulmonary_disease(df, **kwargs):
    """Identifies if a patient has a history of chronic pulmonary disease."""
    pass

def has_connective_tissue_disease(df, **kwargs):
    """Identifies if a patient has a history of connective tissue disease."""
    pass

def has_peptic_ulcer_disease(df, **kwargs):
    """Identifies if a patient has a history of peptic ulcer disease."""
    pass

def has_myocardial_infarction(df, **kwargs):
    """Identifies if a patient has a history of myocardial infarction (MI)."""
    pass

def has_diabetes_without_complications(df, **kwargs):
    """Identifies if a patient has diabetes without chronic complications."""
    pass

def has_diabetes_with_complications(df, **kwargs):
    """Identifies if a patient has diabetes with chronic complications."""
    pass

def has_hemiplegia_or_paraplegia(df, **kwargs):
    """Identifies if a patient has a history of hemiplegia or paraplegia."""
    pass

def has_moderate_to_severe_renal_disease(df, **kwargs):
    """Identifies if a patient has moderate to severe renal disease."""
    pass

def has_malignancy(df, **kwargs):
    """Identifies if a patient has a history of malignancy."""
    pass

def has_moderate_to_severe_liver_disease(df, **kwargs):
    """Identifies if a patient has moderate to severe liver disease."""
    pass

def has_metastatic_solid_tumor(df, **kwargs):
    """Identifies if a patient has a history of a metastatic solid tumor."""
    pass

def has_aids(df, **kwargs):
    """Identifies if a patient has a documented history of AIDS."""
    pass
