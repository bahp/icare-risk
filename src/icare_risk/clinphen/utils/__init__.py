from icare_risk.clinphen.utils.encounters import (
    compute_previous_encounters,
    compute_time_since_last_admission
)

from icare_risk.clinphen.utils.medications import (
    get_qualifying_last_abx
)

from icare_risk.clinphen.utils.vectorized import (
    aggregate_composite_phenotypes,
    compute_phenotype_recency,
    compute_time_since_last_event,
    evaluate_temporal_windows,
    run_vectorized,
)