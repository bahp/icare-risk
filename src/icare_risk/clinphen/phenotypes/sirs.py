from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype
from icare_risk.clinphen.utils.filtering import extract_numeric

@phenotype(
    name="sirs_tachycardia",
    domains=["vitals"],
    category="sirs"
)
def derive_sirs_tachycardia(ctx: EpisodeContext,
                            codes: list,
                            window: tuple = ("0h", "24h"),
                            **kwargs) -> int:
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    hrs = extract_numeric(vitals, "code", "value", codes)

    return 1 if (hrs > 90.0).any() else 0


@phenotype(
    name="sirs_tachypnea",
    domains=["vitals"],
    category="sirs")
def derive_sirs_tachypnea(ctx: EpisodeContext,
                          rr_codes: list,
                          pco2_codes: list = None,
                          window: tuple = ("0h", "24h"),
                          **kwargs) -> int:
    # 1. Check Respiratory Rate
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    rrs = extract_numeric(vitals, "code", "value", rr_codes)
    if (rrs > 20.0).any():
        return 1

    # 2. Check PaCO2 (Safely in case pathology domain isn't loaded)
    if pco2_codes:
        try:
            patho = ctx.get_current("pathology", window=window, columns=["code", "value"])
            pco2s = extract_numeric(patho, "code", "value", pco2_codes)
            if (pco2s < 32.0).any():
                return 1
        except KeyError:
            pass

    return 0


@phenotype(
    name="sirs_abnormal_temp",
    domains=["vitals"],
    category="sirs")
def derive_sirs_abnormal_temp(ctx: EpisodeContext,
                              codes: list,
                              window: tuple = ("0h", "24h"),
                              **kwargs) -> int:
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    temps = extract_numeric(vitals, "code", "value", codes)

    return 1 if ((temps > 38.0) | (temps < 36.0)).any() else 0


@phenotype(
    name="sirs_abnormal_wbc",
    domains=["pathology"],
    category="sirs")
def derive_sirs_abnormal_wbc(
        ctx: EpisodeContext,
        wbc_codes: list,
        bands_codes: list = None,
        wbc_high: float = 12.0,
        wbc_low: float = 4.0,
        bands_thresh: float = 10.0,
        window: tuple = ("0h", "24h"),
        **kwargs
) -> int:
    patho = ctx.get_current("pathology", window=window, columns=["code", "value"])

    # 1. Check primary WBC counts
    wbcs = extract_numeric(patho, "code", "value", wbc_codes)
    if ((wbcs > wbc_high) | (wbcs < wbc_low)).any():
        return 1

    # 2. Check Bandemia if codes are provided
    if bands_codes:
        bands = extract_numeric(patho, "code", "value", bands_codes)
        if (bands > bands_thresh).any():
            return 1

    return 0



# --------------------------------------------------------
# Full rules defined in Yaml
# --------------------------------------------------------
def sirs_tachycardia_score_rule(df, **kwargs):
    """Evaluates patient heart rate to assign points for the SIRS score.

    ??? info "Clinical Definition"
        Systemic Inflammatory Response Syndrome (SIRS) criteria include evaluation 
        for tachycardia. Points are assigned based on heart rate thresholds.

        **Typical Scoring Rubric:**

        | Heart Rate (HR) | Points |
        | :--- | :--- |
        | > 90 bpm | 1 |
        | <= 90 bpm | 0 |

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:sirs_tachycardia_score_rule"
        ```
    """
    pass

def sirs_tachypnea_score_rule(df, **kwargs):
    """Evaluates respiratory rate or pCO2 to assign points for the SIRS score.

    ??? info "Clinical Definition"
        SIRS tachypnea criteria capture elevated respiratory rate or hypocapnia
        as indicators of systemic inflammation.

        **Typical Scoring Rubric:**

        | Parameter | Points |
        | :--- | :--- |
        | Respiratory Rate > 20 breaths/min | 1 |
        | PaCO2 < 32 mmHg | 1 |
        | Normal Parameters | 0 |

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:sirs_tachypnea_score_rule"
        ```
    """
    pass

def sirs_abnormal_temp_score_rule(df, **kwargs):
    """Evaluates body temperature extremes to assign points for the SIRS score.

    ??? info "Clinical Definition"
        SIRS temperature criteria identify acute hyperthermia (fever) or hypothermia.

        **Typical Scoring Rubric:**

        | Temperature Status | Points |
        | :--- | :--- |
        | > 38.0°C or < 36.0°C | 1 |
        | 36.0°C to 38.0°C | 0 |

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:sirs_abnormal_temp_score_rule"
        ```
    """
    pass

def sirs_abnormal_wbc_score_rule(df, **kwargs):
    """Evaluates white blood cell count abnormalities and bandemia for the SIRS score.

    ??? info "Clinical Definition"
        SIRS hematologic criteria capture leukocytosis, leukopenia, or elevated band forms.

        **Typical Scoring Rubric:**

        | Hematologic Marker | Points |
        | :--- | :--- |
        | WBC > 12.0 or < 4.0 x10^3/uL | 1 |
        | Bandemia > 10% | 1 |
        | Normal Range | 0 |

    ??? example "Rule Definition & Parameter Mapping (Click to expand)"

        ```yaml
            --8<-- "src/icare_risk/config/icare/phenotypes.yaml:sirs_abnormal_wbc_score_rule"
        ```
    """
    pass