import pandas as pd
import numpy as np


def compute_previous_encounters(
    df_episodes: pd.DataFrame,
    days: int = None,
    subject_col: str = "SUBJECT",
    spell_col: str = "SPELL_IDENTIFIER",
    admission_date_col: str = "ADMISSION_DATE",
    admission_time_col: str = "ADMISSION_TIME",
) -> pd.DataFrame:
    """Computes the count of prior healthcare encounters per spell.

    If `days` is None, counts all prior encounters in history. If `days` is an
    integer (e.g. 360 or 365), counts encounters within that lookback window.
    """
    df = df_episodes.copy()

    # Create composite admission timestamp if not present
    if "index_admission" not in df.columns:
        df["index_admission"] = pd.to_datetime(
            df[admission_date_col].astype(str)
            + " "
            + df[admission_time_col].astype(str),
            errors="coerce",
        )

    # 1. All-time prior encounters
    if days is None:
        df = df.sort_values(by=[subject_col, "index_admission"])
        df["num_previous_encounters_all_time"] = df.groupby(
            subject_col
        ).cumcount()
        return df

    # 2. Windowed prior encounters (e.g., last 360/365 days)
    col_name = f"num_encounters_last_{days}d"

    # Self-join on subject to compare index admission dates
    merged = pd.merge(
        df[[subject_col, spell_col, "index_admission"]],
        df[[subject_col, spell_col, "index_admission"]],
        on=subject_col,
        suffixes=("", "_prior"),
    )

    # Compute time difference in days
    time_diff_days = (
        merged["index_admission"] - merged["index_admission_prior"]
    ).dt.total_seconds() / 86400.0

    # Filter strictly prior spells within the lookback window
    valid_priors = merged[
        (merged[spell_col] != merged[f"{spell_col}_prior"])
        & (time_diff_days > 0)
        & (time_diff_days <= days)
    ]

    # Aggregate count of unique prior spell identifiers
    counts = (
        valid_priors.groupby([subject_col, spell_col])[f"{spell_col}_prior"]
        .nunique()
        .reset_index(name=col_name)
    )

    out_df = pd.merge(df, counts, on=[subject_col, spell_col], how="left")
    out_df[col_name] = out_df[col_name].fillna(0).astype(int)

    return out_df

import pandas as pd
from pathlib import Path


def compute_time_since_last_admission(
    df_episodes: pd.DataFrame,
    subject_col: str = "SUBJECT",
    spell_col: str = "SPELL_IDENTIFIER",
    admission_col: str = "index_admission",
    discharge_col: str = "discharge_datetime",
    use_discharge_time: bool = True,
) -> pd.DataFrame:
    """Computes time elapsed (in days) since a patient's previous hospital spell

    using only the episodes/admissions cohort dataframe.

    Parameters
    ----------
    df_episodes : pd.DataFrame
        Cohort dataframe containing episode admission (and optionally
        discharge) timestamps.
    use_discharge_time : bool
        If True, computes days from previous DISCHARGE to current admission. If
        False, computes days from previous ADMISSION to current admission.
    """
    df = df_episodes.copy()
    df[admission_col] = pd.to_datetime(df[admission_col])

    # Reference timestamp to evaluate against (prior discharge or prior admission)
    prior_time_col = (
        discharge_col
        if (use_discharge_time and discharge_col in df.columns)
        else admission_col
    )
    df[prior_time_col] = pd.to_datetime(df[prior_time_col])

    # Sort strictly by patient and admission time for merge_asof
    episodes_sorted = df[[subject_col, spell_col, admission_col]].sort_values(
        admission_col
    )

    # Historical timeline of prior spells
    history = df[[subject_col, prior_time_col]].dropna(subset=[prior_time_col]).sort_values(prior_time_col)

    # Match each episode to the most recent prior spell strictly before current admission
    merged = pd.merge_asof(
        episodes_sorted,
        history,
        by=subject_col,
        left_on=admission_col,
        right_on=prior_time_col,
        direction="backward",
        allow_exact_matches=False,  # strictly prior (< index_admission)
    )

    # Compute days elapsed
    metric_col = (
        "days_since_last_discharge"
        if use_discharge_time
        else "days_since_last_admission"
    )
    merged[metric_col] = (
        merged[admission_col] - merged[prior_time_col]
    ).dt.total_seconds() / 86400.0

    # Join result back onto base cohort
    result = pd.merge(
        df_episodes,
        merged[[subject_col, spell_col, metric_col]],
        on=[subject_col, spell_col],
        how="left",
    )

    return result
