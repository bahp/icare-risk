from icare_risk.core.registry import register_phenotype
from icare_risk.core.context import EpisodeContext

@register_phenotype("history_of_diabetes")
def history_of_diabetes(ctx: EpisodeContext) -> bool:
    prob_conf = ctx.schema.problems
    rx_conf = ctx.schema.prescribing

    # Check prior problem lists
    probs = ctx.get_past("problems", columns=[prob_conf.code_col])
    if not probs.empty and probs[prob_conf.code_col].astype(str).str.startswith("E11").any():
        return True

    # Check prior prescriptions (Pushing down Metformin via string filtering after fetch)
    # Alternatively, if we had standardized RXNorm codes, we could use concept_codes=[] pushdown
    rxs = ctx.get_past("prescribing", columns=[rx_conf.code_col])
    if not rxs.empty and rxs[rx_conf.code_col].astype(str).str.contains("Metformin", case=False, na=False).any():
        return True

    return False