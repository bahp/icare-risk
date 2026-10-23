import pandas as pd
from icare_risk.core.registry import register_phenotype
from icare_risk.core.context import EpisodeContext

def compute_hypoxia(vitals_df: pd.DataFrame, val_col: str, threshold: float = 92.0) -> bool:
    if vitals_df.empty: return False
    vals = pd.to_numeric(vitals_df[val_col], errors='coerce')
    return (vals < threshold).any()

@register_phenotype("is_hypoxic_first_24h")
def is_hypoxic_first_24h(ctx: EpisodeContext) -> bool:
    conf = ctx.schema.vitals
    # Note the code pushdown: we only fetch LOINC-2708-6 (Oxygen Saturation)
    df = ctx.get_window(
        "vitals", start_h=0, end_h=24,
        columns=[conf.code_col, conf.value_col],
        concept_codes=["LOINC-2708-6"]
    )
    return compute_hypoxia(df, conf.value_col)