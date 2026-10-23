import pandas as pd
from icare_risk.core.registry import register_phenotype
from icare_risk.core.context import EpisodeContext


@register_phenotype("early_aki_risk")
def early_aki_risk(ctx: EpisodeContext) -> bool:
    """
    Detects Acute Kidney Injury (AKI) risk:
    Creatinine rose by >= 0.3 mg/dL or >= 50% from historical baseline in the first 48h,
    AND the patient is actively on Vancomycin.
    """
    path_conf = ctx.schema.pathology
    rx_conf = ctx.schema.prescribing

    # 1. Get baseline (historical) creatinine
    # Pushdown: Only fetch LOINC-2160-0 (Creatinine) prior to admission
    past_labs = ctx.get_past(
        "pathology",
        columns=[path_conf.code_col, path_conf.value_col],
        concept_codes=["LOINC-2160-0"]
    )

    if past_labs.empty:
        baseline_cr = None
    else:
        baseline_cr = pd.to_numeric(past_labs[path_conf.value_col], errors='coerce').min()

    # 2. Get peak creatinine in first 48h of admission
    window_labs = ctx.get_window(
        "pathology",
        start_h=0, end_h=48,
        columns=[path_conf.code_col, path_conf.value_col],
        concept_codes=["LOINC-2160-0"]
    )

    if window_labs.empty:
        peak_cr = None
    else:
        peak_cr = pd.to_numeric(window_labs[path_conf.value_col], errors='coerce').max()

    # 3. Check active Vancomycin in first 48h
    rxs = ctx.get_window(
        "prescribing",
        start_h=0, end_h=48,
        columns=[rx_conf.code_col]
    )

    has_vanc = False
    if not rxs.empty:
        # Check if any medication name contains "Vancomycin"
        has_vanc = rxs[rx_conf.code_col].astype(str).str.contains("Vancomycin", case=False, na=False).any()

    # 4. Compute AKI clinical criteria
    if pd.notna(baseline_cr) and pd.notna(peak_cr):
        aki_criteria = (peak_cr >= baseline_cr + 0.3) or (peak_cr >= baseline_cr * 1.5)
        return bool(aki_criteria and has_vanc)

    return False