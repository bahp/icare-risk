"""Current-episode, windowed phenotypes."""
from __future__ import annotations

import numpy as np

from ..registry.registry import phenotype
from ..temporal.context import EpisodeContext


@phenotype(
    name="min_spo2_24h",
    domains=["vitals"],
    category="windowed",
    description="Minimum SpO2 recorded in the first 24h of the index admission.",
)
def min_spo2_24h(ctx: EpisodeContext, code: str = None, window: tuple = ("0h", "24h"), **kwargs) -> float:
    print("INSIDE FUNCTION")
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    print(vitals)
    if vitals.empty:
        return np.nan

    # Uses the dynamically passed code parameter instead of a hardcoded string
    spo2 = vitals.loc[vitals["code"].astype(str).str.upper() == str(code), "value"]
    return float(spo2.min()) if spo2.notna().any() else np.nan

@phenotype(
    name="min_temp_24h",
    domains=["vitals"],
    category="windowed", # baseline
    description="",
)
def min_temp_24h(ctx: EpisodeContext, code, window, **kwargs) -> float:
    vitals = ctx.get_current("vitals",
        window=("0h", "24h"), columns=["code", "value"])
    print("TEMP")
    if vitals.empty:
        return np.nan
    temp = vitals.loc[vitals["code"].astype(str).str.upper() == code, "value"]
    return float(temp.min()) if temp.notna().any() else np.nan



