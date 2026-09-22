import pandas as pd

from icare_risk.clinphen.utils.filtering import match_codes
from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype


# -----------------------------------------------------------------------------------
#                 HOLMGREN SCORE EXTRACTORS
# -----------------------------------------------------------------------------------

# --8<-- [start:holmgren_hx_hospital_care_abroad]
def holmgren_hx_hospital_care_abroad(df, **kwargs):
    """Identifies if a patient was hospitalized abroad in the last 12 months.

    ??? info "Clinical Definition"
        Evaluates encounter history or international healthcare travel indicators within an 8,760-hour window prior to index.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:holmgren_hx_hospital_care_abroad"
        ```
    """
    pass
# --8<-- [end:holmgren_hx_hospital_care_abroad]


# --8<-- [start:holmgren_hx_previous_3gcr_culture]
def holmgren_hx_previous_3gcr_culture(df, **kwargs):
    """Identifies if a patient has a history of previous third-generation cephalosporin-resistant (3GCR) culture in blood or urine.

    ??? info "Clinical Definition"
        Evaluates microbiology or problem history for documented prior 3GCR isolates from blood or urine specimens.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:holmgren_hx_previous_3gcr_culture"
        ```
    """
    pass
# --8<-- [end:holmgren_hx_previous_3gcr_culture]


# --8<-- [start:holmgren_hx_previous_3gcr_rectal_swab]
def holmgren_hx_previous_3gcr_rectal_swab(df, **kwargs):
    """Identifies if a patient has a history of previous third-generation cephalosporin-resistant (3GCR) detection in a rectal swab.

    ??? info "Clinical Definition"
        Evaluates microbiology, screening, or surveillance data for prior 3GCR carriage detected via rectal swabs.

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"
        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:holmgren_hx_previous_3gcr_rectal_swab"
        ```
    """
    pass
# --8<-- [end:holmgren_hx_previous_3gcr_rectal_swab]