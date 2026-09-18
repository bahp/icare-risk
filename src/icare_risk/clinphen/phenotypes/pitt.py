# Libraries
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, Any

from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype

"""
Notes
-----
1. Think about creating a generic evaluate score rules in which you can
   pass a set of rules, with the corresponding number of points to add
   and it evaluates it, so that it can be reused.
   
   Note that strings might be cleaned differently than numbers.

"""

def compute_pitt_fever(temps: pd.Series) -> pd.Series:
    temps = pd.to_numeric(temps, errors='coerce')
    conditions = [
        (temps <= 35.0) | (temps >= 40.0),
        (temps >= 35.1) | (temps <= 36.0),
        (temps >= 39.0) & (temps <= 39.9)
    ]
    choices = [2, 1, 1]
    points = np.select(conditions, choices, default=0)
    return pd.Series(points, index=temps.index)


def compute_pitt_mental(gcs: pd.Series) -> pd.Series:
    gcs = pd.to_numeric(gcs, errors='coerce')
    conditions = [
        (gcs <= 9),
        (gcs >= 10) & (gcs <= 12),
        (gcs >= 13) & (gcs <= 14)
    ]
    choices = [4, 2, 1]
    points = np.select(conditions, choices, default=0)
    return pd.Series(points, index=gcs.index)


# --------------------------------------------------------
# Helper phenotypes
# --------------------------------------------------------
def is_mechanically_ventilated(dataset):
    """Determines whether it has mechanically ventilated support.

    Clinical Logic
    --------------
    - Receiving mechanical ventilation: +2 pts
    - Search respiratory support flowsheets.

    Returns
    -------
    pd.Series (int)
        1 if ventilated, 0 otherwise.
    """
    pass

def has_recent_cardiac_arrest(dataset):
    """Checks whether it has cardiac arrest.

     Clinical Logic
     --------------
    - Cardiac arrest within window: +4 pts
    - Check resuscitation event codes.

    Returns
    -------
    pd.Series (int)
        1 if cardiac arrest, 0 otherwise.
    """
    pass


def has_vasopressors(dataset, window_hours=24, target_meds=[]):
    """Determines whether a patient has vasopressors."""
    pass


# --------------------------------------------------------
# Score functions
# --------------------------------------------------------
import numpy as np
from icare_risk.clinphen.temporal.context import EpisodeContext
from icare_risk.clinphen.registry.registry import phenotype
from icare_risk.clinphen.utils.filtering import match_codes, extract_numeric


@phenotype(
    name="pitt_fever_score",
    domains=["vitals"],
    category="pitt_components"
)
def compute_pitt_fever_score(ctx: EpisodeContext,
                             codes: list,
                             window: tuple = ("0h", "24h"),
                             **kwargs) -> int:
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    temps = extract_numeric(vitals, "code", "value", codes)
    if temps.empty: return 0
    scores = compute_pitt_fever(temps)
    return int(scores.max())


@phenotype(
    name="pitt_mental_score",
    domains=["vitals"],
    category="pitt_components")
def compute_pitt_mental_score(ctx: EpisodeContext,
                              codes: list,
                              window: tuple = ("0h", "24h"),
                              **kwargs) -> int:
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    gcs = extract_numeric(vitals, "code", "value", codes)

    if gcs.empty: return 0
    conditions = [
        (gcs <= 9),
        (gcs >= 10) & (gcs <= 12),
        (gcs >= 13) & (gcs <= 14)
    ]
    return int(np.select(conditions, [4, 2, 1], default=0).max())


@phenotype(
    name="pitt_hypo_score",
    domains=["vitals"], #, "prescribing"],
    category="pitt_components"
)
def compute_pitt_hypo_score(ctx: EpisodeContext,
                            sbp_codes: list,
                            vasopressor_drugs: list,
                            window: tuple = ("0h", "24h"),
                            **kwargs) -> int:
    return 0

@phenotype(
    name="pitt_vent_score",
    domains=[],
    category="pitt_components"
)
def compute_pitt_vent_score(ctx: EpisodeContext,
                            codes: list,
                            window: tuple = ("0h", "24h"),
                            **kwargs) -> int:
    return 0


@phenotype(
    name="pitt_arrest_score",
    domains=[],
    category="pitt_components"
)
def compute_pitt_arrest_score(ctx: EpisodeContext,
                              codes: list,
                              window: tuple = ("0h", "24h"),
                              **kwargs) -> int:
    return 0


@phenotype(
    name="pitt_bacteremia_score",
    domains=["vitals", "prescribing"],#, "interventions"],
    category="clinical_scores"
)
def compute_pitt_score(
        ctx: EpisodeContext,
        fever_codes: list,
        gcs_codes: list,
        sbp_codes: list,
        vasopressor_drugs: list,
        vent_codes: list,
        arrest_codes: list,
        window: tuple = ("0h", "24h"),
        **kwargs
) -> dict:
    """Calculates all Pitt components for a single episode.

    Note: Do not use.
    """

    # --- 1. Fever Score ---
    fever_score = 0
    vitals = ctx.get_current("vitals", window=window, columns=["code", "value"])
    if not vitals.empty:
        temp_mask = vitals["code"].astype(str).str.upper().isin([str(c).upper() for c in fever_codes])
        temps = pd.to_numeric(vitals.loc[temp_mask, "value"], errors="coerce").dropna()
        if not temps.empty:
            def score_temp(t):
                if t <= 35.0 or t >= 40.0: return 2
                if (35.1 <= t <= 36.0) or (39.0 <= t <= 39.9): return 1
                return 0

            fever_score = int(temps.apply(score_temp).max())

    # --- 2. Mental Score (GCS) ---
    mental_score = 0
    if not vitals.empty:
        gcs_mask = vitals["code"].astype(str).str.upper().isin([str(c).upper() for c in gcs_codes])
        gcs_vals = pd.to_numeric(vitals.loc[gcs_mask, "value"], errors="coerce").dropna()
        if not gcs_vals.empty:
            def score_gcs(g):
                if g <= 9: return 4
                if 10 <= g <= 12: return 2
                if 13 <= g <= 14: return 1
                return 0

            mental_score = int(gcs_vals.apply(score_gcs).max())

    # --- 3. Hypotension Score ---
    hypo_score = 0
    # Check SBP
    if not vitals.empty:
        sbp_mask = vitals["code"].astype(str).str.upper().isin([str(c).upper() for c in sbp_codes])
        sbps = pd.to_numeric(vitals.loc[sbp_mask, "value"], errors="coerce").dropna()
        if not sbps.empty and (sbps < 90.0).any():
            hypo_score = 2

    # Check Vasopressors (if SBP didn't already trigger the 2 points)
    #if hypo_score == 0:
    #    rx = ctx.get_current("prescribing", window=window, columns=["drug"])
    #    if not rx.empty:
    #        rx_mask = rx["drug"].astype(str).str.upper().isin([str(d).upper() for d in vasopressor_drugs])
    #        if rx_mask.any():
    #            hypo_score = 2

    # --- 4. Ventilation & Arrest Scores ---
    vent_score = 0
    arrest_score = 0
    #interventions = ctx.get_current("interventions", window=window, columns=["code"])
    #if not interventions.empty:
    #    code_series = interventions["code"].astype(str).str.upper()
    #    if code_series.isin([str(c).upper() for c in vent_codes]).any():
    #        vent_score = 2
    #    if code_series.isin([str(c).upper() for c in arrest_codes]).any():
    #        arrest_score = 4

    # --- Return Dictionary ---
    # runner.py will automatically prefix these keys with "pitt_bacteremia_score__"
    total = fever_score + mental_score + hypo_score + vent_score + arrest_score
    return {
        "fever": fever_score,
        "mental": mental_score,
        "hypo": hypo_score,
        "vent": vent_score,
        "arrest": arrest_score,
        "total": total
    }