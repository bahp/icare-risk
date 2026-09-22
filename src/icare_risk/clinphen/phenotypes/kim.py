# -----------------------------------------------------------------------------------
#                 KIM SCORE EXTRACTORS
# -----------------------------------------------------------------------------------
# These methods extract the clinical variables required to compute the Kim score
# for predicting extended-spectrum beta-lactamase (ESBL) producing Enterobacteriaceae.

def kim_hx_prior_esbl(df, **kwargs):
    """Identifies if a patient has a documented history of prior ESBL colonization or infection.

    ??? info "Clinical Definition"
        We consider a patient to have prior ESBL if:
        1. Explicit History: An ESBL-specific code or organism flag appears in 'problems' or historical microbiology.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:kim_hx_prior_esbl"
        ```
    """
    pass


def kim_hx_recent_hosp_1yr(df, **kwargs):
    """Identifies if a patient had a hospital admission within the last 1 year.

    ??? info "Clinical Definition"
        Evaluates encounters or historical admissions within a 365-day
        window prior to index.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:kim_hx_recent_hosp_1yr"
        ```
    """
    pass


def kim_hx_nursing_home_resident(df, **kwargs):
    """Identifies if a patient is a resident in a long-term care facility or nursing home.

    ??? info "Clinical Definition"
        Captured via structured problem list or diagnosis code (e.g., ICD-10 Z59.3).

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:kim_hx_nursing_home_resident"
        ```
    """
    pass


def kim_hx_urinary_catheter(df, **kwargs):
    """Identifies the presence or use of an indwelling urinary catheter.

    ??? info "Clinical Definition"
        Captured via device codes, procedure history, or urinary device encounter codes.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:kim_hx_urinary_catheter"
        ```
    """
    pass


def kim_hx_prior_antibiotics_90d(df, **kwargs):
    """Determines if a patient has a history of antibiotic use within the prior 90 days.

    ??? info "Clinical Definition"
        Evaluates the prescribing table for systemic antibiotic administration over a 2,160-hour window.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:kim_hx_prior_antibiotics_90d"
        ```
    """
    pass