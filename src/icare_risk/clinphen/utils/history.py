# Libraries
import pandas as pd

from typing import Dict
from icare_risk.clinphen.utils.filtering import match_codes

from typing import Dict, List, Set, Any

from typing import Any, Dict, List, Set

def extract_all_codes(
    config: Any,
    all_configs: Dict[str, Any] = None,
    visited: Set[int] = None
) -> List[str]:
    """
    Recursively extracts all clinical codes from a phenotype configuration.
    Handles:
    - Direct keys: 'codes', 'code', 'icd10', 'snomed', 'code_set', 'patterns'
    - Nested structures: kwargs, components, sub-rules
    - Cross-references: composite components pointing to another phenotype name
    """
    if visited is None:
        visited = set()

    found_codes: Set[str] = set()

    # Prevent infinite loops on cyclic references
    config_id = id(config)
    if config_id in visited:
        return []
    visited.add(config_id)

    # Keys commonly used to store clinical codes across different YAML formats
    CODE_KEYS = {"codes", "code", "icd10", "icd10_codes", "snomed", "snomed_codes", "code_set", "patterns"}

    if isinstance(config, dict):
        # 1. Handle phenotype cross-references (e.g., component referencing another phenotype)
        ref_name = config.get("phenotype") or config.get("ref") or config.get("spec_name")
        if ref_name and all_configs and ref_name in all_configs:
            found_codes.update(extract_all_codes(all_configs[ref_name], all_configs, visited))

        # 2. Iterate keys
        for key, val in config.items():
            if key.lower() in CODE_KEYS:
                if isinstance(val, (list, tuple, set)):
                    found_codes.update(str(c) for c in val if c is not None)
                elif isinstance(val, (str, int)):
                    found_codes.add(str(val))
            else:
                # Recurse into nested dicts/lists (kwargs, components, logic rules)
                found_codes.update(extract_all_codes(val, all_configs, visited))

    elif isinstance(config, (list, tuple, set)):
        for item in config:
            found_codes.update(extract_all_codes(item, all_configs, visited))

    return list(found_codes)


def build_historical_events_table(
    df: pd.DataFrame,
    configs: Dict[str, dict],
    subject_col: str = "subject",
    time_col: str = "timestamp",
    code_col: str = "code",
    config_code_key: str = "codes"
) -> pd.DataFrame:
    """Builds a generic historical events table tracking the first occurrence date per subject and event type.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing raw event data, subjects, and codes.
    configs : Dict[str, dict]
        A dictionary mapping event names to their respective configuration dictionaries.
    subject_col : str, optional
        The column name representing the subject identifier, by default "subject".
    time_col : str, optional
        The column name representing the event timestamp, by default "timestamp".
    code_col : str, optional
        The column name representing the event code, by default "code".
    config_code_key : str, optional
        The key used to extract code lists from the configuration kwargs, by default "codes".

    Returns
    -------
    pd.DataFrame
        A table containing columns `[subject_col, "event_name", "first_occurrence_date"]`.
    """
    if df.empty or not configs:
        return pd.DataFrame(columns=[subject_col,
            "event_name", "first_occurrence_date"])

    # Get the column with the codes and convert to str
    code_series = df[code_col].fillna('').astype(str)

    # Apply match_codes for each YAML config directly to the domain dataframe
    for name, config in configs.items():
        #codes = config.get("kwargs", {}).get(config_code_key, [])
        codes = extract_all_codes(config, all_configs=configs)
        df[name] = match_codes(code_series, codes)

    # Melt to isolate positive matches
    melted = df.melt(
        id_vars=[subject_col, time_col],
        value_vars=list(configs.keys()),
        var_name="event_name",
        value_name="is_present"
    )

    # Find the very first occurrence timestamp per patient and condition
    positive_events = melted[melted["is_present"]]
    first_occurrences = positive_events \
        .groupby([subject_col, "event_name"])[time_col] \
        .min().reset_index()
    first_occurrences.rename(columns={time_col: "first_occurrence_date"}, inplace=True)

    return first_occurrences

def evaluate_temporal_phenotypes(
    episodes_df: pd.DataFrame,
    first_occurrences: pd.DataFrame,
    subject_col: str = "subject",
    admission_col: str = "index_admission",
    event_col: str = "event_name",
    time_col: str = "first_occurrence_date",
    prefix: str = "hx_"
) -> pd.DataFrame:
    """Evaluates the temporal boundary for all patient episodes at once.

    Checks whether historical events occurred on or before the index admission date,
    generating binary indicator columns for each phenotype.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        The input DataFrame containing patient episodes and index admission dates.
    first_occurrences : pd.DataFrame
        The DataFrame containing first occurrence timestamps per subject and event.
    subject_col : str, optional
        The column name representing the subject identifier, by default "subject".
    admission_col : str, optional
        The column name representing the index admission timestamp, by default "index_admission".
    event_col : str, optional
        The column name representing the event name, by default "event_name".
    time_col : str, optional
        The column name representing the first occurrence date, by default "first_occurrence_date".
    prefix : str, optional
        The prefix to prepend to generated binary indicator feature columns, by default "hx_".

    Returns
    -------
    pd.DataFrame
        A copy of `episodes_df` containing new binary indicator columns (named
        according to `prefix` + event name) denoting whether each event occurred
        prior to or at the index admission.
    """
    if first_occurrences.empty:
        return episodes_df.copy()

    # Pivot so each event becomes a column with the first occurrence date[cite: 1]
    patient_history = first_occurrences.pivot(
        index=subject_col,
        columns=event_col,
        values=time_col
    ).reset_index()

    # Join to the full episodes cohort[cite: 1]
    merged = episodes_df.merge(patient_history, on=subject_col, how="left")
    event_names = first_occurrences[event_col].unique()

    # Vectorized temporal check[cite: 1]
    for event in event_names:
        target_col = f"{prefix}{event}"
        valid_dates = merged[event].fillna(pd.Timestamp.max)
        merged[target_col] = (valid_dates <= merged[admission_col]).astype(int)

    # NEW CLEANUP STEP: Drop the intermediate date columns to prevent downstream data leakage
    # We only drop them if the prefix didn't already overwrite them
    cols_to_drop = [e for e in event_names if f"{prefix}{e}" != e]
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)

    return merged
