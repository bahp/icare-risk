import pandas as pd

from icare_risk.clinphen.utils.filtering import match_codes
from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype



# -----------------------------------------------------------------------------------
#                                 Other Definitions
# -----------------------------------------------------------------------------------
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

    !!! tip "Multimodal"

    ??? info "Clinical Definition"
        We consider a patient to have CHF if ANY of the following are true:

        1. Explicit History: The `diabetes` code appears in 'problems'.
        2. Explicit Diagnosis: The `diabetes' code appears in 'diagnosis` -> Not implemented.
        3. Medication Proxy: The `prescribing` table contains ['insulin', 'metformin', 'gliclazide'].
        4. Lab Values Proxy: The rolling maximum glucose (`glucose_max_24h`) is > 200 mg/dL.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:has_diabetes"
        ```

    Returns
    -------
    pd.Series of integers (1 for has_diabetes, 0 for no diabetes).
    """
    pass




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

def charlson_hx_chf(df, **kwargs):
    """
    Determines if a patient has a history of Congestive Heart Failure (CHF).

    <!-- icare_table: charlson_hx_chf -->


    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_chf"
        ```

        --8<-- "docs/_snippets/charlson_hx_chf.md"

    ??? info "Clinical Definition"
        We consider a patient to have CHF if ANY of the following are true:

        1. Explicit History: The 'heart failure' code appears in 'problems'.
        2. Explicit Diagnosis: The 'heart failure' code appears in 'diagnosis` -> Not implemented.
        3. Medication Proxy: The patient is actively prescribed specific heart failure
           medications like 'entresto', 'milrinone', or 'dobutamine' in 'prescribing' current
           episode or stay.
    """

def charlson_hx_pvd(df, **kwargs):
    """Identifies if a patient has a history of peripheral vascular disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_pvd"
        ```

        --8<-- "docs/_snippets/charlson_hx_pvd.md"
    """
    pass

def charlson_hx_stroke(df, **kwargs):
    """Identifies if a patient has a history of cerebrovascular disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_stroke"
        ```

        --8<-- "docs/_snippets/charlson_hx_stroke.md"
    """
    pass

def charlson_hx_dementia(df, **kwargs):
    """Identifies if a patient has a documented history of dementia.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_dementia"
        ```

        --8<-- "docs/_snippets/charlson_hx_dementia.md"
    """
    pass

def charlson_hx_pulmonary(df, **kwargs):
    """Identifies if a patient has a history of chronic pulmonary disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_pulmonary"
        ```

        --8<-- "docs/_snippets/charlson_hx_pulmonary.md"
    """
    pass

def charlson_hx_rheum(df, **kwargs):
    """Identifies if a patient has a history of connective tissue disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_rheum"
        ```

        --8<-- "docs/_snippets/charlson_hx_rheum.md"
    """
    pass

def charlson_hx_pud(df, **kwargs):
    """Identifies if a patient has a history of peptic ulcer disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_pud"
        ```

        --8<-- "docs/_snippets/charlson_hx_pud.md"
    """
    pass

def charlson_hx_mi(df, **kwargs):
    """Identifies if a patient has a history of myocardial infarction (MI).

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_mi"
        ```

        --8<-- "docs/_snippets/charlson_hx_mi.md"
    """
    pass

def charlson_hx_liver_mild(df, **kwargs):
    """Determines if a patient has mild liver disease.

    !!! tip "Multimodal"

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:has_mild_liver_disease"
        ```

        --8<-- "docs/_snippets/charlson_hx_liver_mild.md"


    ??? info "Clinical Definition"
        We consider a patient to have mild liver disease if ANY of the following are true:

        1. Explicit History: The `mild liver disease' code appears in 'problems'.
        2. ICD-10 Codes:
        3. Medication Proxy: The `prescribing` table contains ['lactulose'].
        4. Lab Values Proxy: The 'pathology' table contains (Bilirubin > 2.0)
    """
    pass

def charlson_hx_diabetes_uncomp(df, **kwargs):
    """Identifies if a patient has diabetes without chronic complications.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_diabetes_uncomp"
        ```

        --8<-- "docs/_snippets/charlson_hx_diabetes_uncomp.md"
    """
    pass

def charlson_hx_diabetes_comp(df, **kwargs):
    """Identifies if a patient has diabetes with chronic complications.

    !!! warning "Pending to be properly defined!"

    ??? info "Clinical Definition"
        Refers to diabetes accompanied by chronic end-organ damage (often referred
        to as diabetic end-organ damage). This includes diabetic microvascular or
        macrovascular complications such as retinopathy, neuropathy, nephropathy,
        or diabetic angiopathy (excluding simple uncomplicated routine management).

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_diabetes_comp"
        ```

        --8<-- "docs/_snippets/charlson_hx_diabetes_comp.md"

    """
    pass

def charlson_hx_hemiplegia(df, **kwargs):
    """Identifies if a patient has a history of hemiplegia or paraplegia.

    !!! warning "Pending problem codes definitions"

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_hemiplegia"
        ```
    """
    pass

def charlson_hx_renal_mod_sev(df, **kwargs):
    """Identifies if a patient has moderate to severe renal disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_renal_mod_sev"
        ```

        --8<-- "docs/_snippets/charlson_hx_renal_mod_sev.md"
    """
    pass

def charlson_hx_liver_mod_sev(df, **kwargs):
    """Identifies if a patient has moderate to severe liver disease.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_liver_mod_sev"
        ```

        --8<-- "docs/_snippets/charlson_hx_liver_mod_sev.md"
    """
    pass

def charlson_hx_cancer_solid(df, **kwargs):
    """Identifies if a patient has a history of a non-metastatic solid tumor.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_cancer_solid"
        ```

        --8<-- "docs/_snippets/charlson_hx_cancer_solid.md"
    """
    pass

def charlson_hx_cancer_met(df, **kwargs):
    """Identifies if a patient has a history of a metastatic solid tumor.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_cancer_met"
        ```

        --8<-- "docs/_snippets/charlson_hx_cancer_met.md"
    """
    pass

def charlson_hx_aids(df, **kwargs):
    """Identifies if a patient has a documented history of AIDS.

    !!! warning "Pending problem codes definitions"

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_aids"
        ```
    """
    pass

def charlson_hx_hiv(df, **kwargs):
    """Identifies if a patient has a documented history of HIV.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_hiv"
        ```
    """
    pass

def charlson_hx_leukemia(df, **kwargs):
    """Identifies if a patient has a documented history of leukemia.

    !!! warning "Pending problem codes definitions"

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:charlson_hx_leukemia"
        ```
    """
    pass