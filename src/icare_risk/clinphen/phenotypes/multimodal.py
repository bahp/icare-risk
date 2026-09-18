from typing import Any, Dict, List, Optional
from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype
from icare_risk.clinphen.phenotypes.utils import (
    derive_from_history_code,
    derive_from_keyword,
    derive_from_expression
)

@phenotype(
    name="multimodal_phenotype",
    domains=["problems", "prescribing", "pathology"],
    category="generic"
)
def derive_multimodal_phenotype(
    ctx: EpisodeContext,
    code_rules: Optional[List[Dict[str, Any]]] = None,
    keyword_rules: Optional[List[Dict[str, Any]]] = None,
    expression_rules: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> int:
    """
    Evaluates compound clinical criteria across multiple domains using logical OR.

    Note
    ----
    Can probably be deprecated in favor of utils.derive_composite_rules

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episodic data[cite: 4].
    code_rules : List[Dict[str, Any]], optional
        List of keyword argument dictionaries to forward to `derive_from_history_code`[cite: 1].
    keyword_rules : List[Dict[str, Any]], optional
        List of keyword argument dictionaries to forward to `derive_from_keyword`.
    expression_rules : List[Dict[str, Any]], optional
        List of keyword argument dictionaries to forward to `derive_from_expression`.
    **kwargs
        Additional keyword arguments absorbed from the phenotype specification.

    Returns
    -------
    int
        1 if any rule evaluates to True; otherwise 0.
    """
    # 1. Evaluate structured diagnostic / problem codes
    if code_rules:
        for rule in code_rules:
            if derive_from_history_code(ctx, **rule) == 1:
                return 1

    # 2. Evaluate free-text medications or descriptions
    if keyword_rules:
        for rule in keyword_rules:
            if derive_from_keyword(ctx, **rule) == 1:
                return 1

    # 3. Evaluate numeric laboratory thresholds or ranges via query expressions
    if expression_rules:
        for rule in expression_rules:
            if derive_from_expression(ctx, **rule) == 1:
                return 1

    return 0