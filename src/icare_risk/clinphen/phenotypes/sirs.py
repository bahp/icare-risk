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