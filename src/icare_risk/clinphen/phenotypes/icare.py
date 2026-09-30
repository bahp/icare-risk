# Libraries
import pandas as pd

from typing import Any, Sequence


def derive_charlson_history_rules(
    df: pd.DataFrame,
    snomed_codes: Sequence[str] | None = None,
    res195_codes: Sequence[str] | None = None,
    window: tuple[Any, str] | list[Any] = (None, "24h"),
    logic: str = "max",
    **kwargs,
) -> pd.Series:
    """Builds standard Charlson history rules across problems and diagnosis domains.

    Automatically constructs components for:
    1. Problems domain (SNOMED CT)
    2. Diagnosis domain (ICD-10 / RES195) with specified time window
    3. Diagnosis domain (SNOMED CT) with specified time window
    """
    from icare_risk.clinphen.phenotypes.utils import derive_composite_rules

    components = []
    window_list = list(window) if isinstance(window, (list, tuple)) else window

    # 1. Problems history (SNOMED CT)
    if snomed_codes:
        components.append({
            "extractor_type": "history",
            "domain": "problems",
            "codes": list(snomed_codes),
        })

    # 2. Diagnosis history (ICD-10 / RES195)
    if res195_codes:
        components.append({
            "extractor_type": "history",
            "domain": "diagnosis",
            "codes": list(res195_codes),
            "code_col": "DIAGNOSIS_CODE_ICD",
            "window": window_list,
        })

    # 3. Diagnosis history (SNOMED CT)
    if snomed_codes:
        components.append({
            "extractor_type": "history",
            "domain": "diagnosis",
            "codes": list(snomed_codes),
            "code_col": "DIAGNOSIS_CODE_SNOMED",
            "window": window_list,
        })

    return derive_composite_rules(df, logic=logic, components=components, **kwargs)