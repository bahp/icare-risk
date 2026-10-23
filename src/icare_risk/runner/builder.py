import pandas as pd
import duckdb
from typing import List, Optional
from icare_risk.core.registry import PHENOTYPE_REGISTRY
from icare_risk.core.schema import ClinicalSchema
from icare_risk.core.context import EpisodeContext
from icare_risk.data.dataset import Dataset


def build_feature_matrix(
        cohort_df: pd.DataFrame,
        dataset: Dataset,
        schema: ClinicalSchema,
        phenotypes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Iterates over a cohort DataFrame, computes registered phenotypes,
    and returns the final feature matrix.
    """
    # Filter to only the requested phenotypes (or run all of them)
    funcs_to_run = {
        name: func for name, func in PHENOTYPE_REGISTRY.items()
        if (phenotypes is None) or (name in phenotypes)
    }

    results = []

    # Extract column mappings dynamically from schema
    subj_col = schema.episodes.subject_col
    enc_col = schema.episodes.encounter_col
    adm_dt_col = schema.episodes.timestamp_col

    for idx, row in cohort_df.iterrows():
        adm_dt = pd.to_datetime(row[adm_dt_col])

        # Parse discharge date if your dataset has it (needed for temporal bounding)
        # Adjust 'DISCHARGE_DATE' to whatever your schema calls it
        dis_dt = pd.to_datetime(row['DISCHARGE_DATE']) if 'DISCHARGE_DATE' in row else None

        # Safely assign an encounter ID. If the column doesn't exist, generate a surrogate key.
        if enc_col and enc_col in row:
            enc_id = row[enc_col]
        else:
            enc_id = f"stay_{idx}"  # Surrogate key

        ctx = EpisodeContext(
            subject_id=row[subj_col],
            encounter_id=enc_id,
            admission_dt=adm_dt,
            discharge_dt=dis_dt,
            dataset=dataset,
            schema=schema
        )

        stay_features = {
            "SUBJECT": row[subj_col],
            "ENCNTR_ID": enc_id  # Matrix will output stay_0, stay_1, etc.
        }

        for name, func in funcs_to_run.items():
            stay_features[name] = func(ctx)

        results.append(stay_features)

    return pd.DataFrame(results)