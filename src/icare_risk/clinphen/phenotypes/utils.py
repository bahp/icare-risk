# Libraries
import operator
import pandas as pd
from typing import List, Tuple

from icare_risk.clinphen.utils.filtering import match_codes
from icare_risk.clinphen.temporal.context import EpisodeContext


def derive_from_expression(
        ctx: EpisodeContext,
        domain: str,
        codes: list,
        query_expr: str,
        value_col: str = "value",
        code_col: str = "code",
        check_current: bool = True,
        window: tuple = ("0h", "24h"),
        **kwargs
) -> int:
    """
    Evaluates a pandas-compatible query string against numeric clinical values.

    This extractor filters records matching specific clinical codes and evaluates
    their corresponding numeric values using flexible string expressions
    (e.g., '150 <= value <= 250' or 'value > 200').

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data[cite: 4].
    domain : str
        The clinical domain table to query (e.g., 'labs', 'vitals')[cite: 4].
    codes : list
        A list of target clinical codes to filter the domain data before evaluating the expression.
    query_expr : str
        A valid pandas query expression string referencing the `value_col` (e.g., 'value < 4.0').
    value_col : str, optional
        The dataframe column containing the numeric values to evaluate, by default "value".
    code_col : str, optional
        The dataframe column containing the clinical codes to match against, by default "code".
    check_current : bool, optional
        If True, evaluates records within the specified current admission window[cite: 4].
        If False, evaluates strictly historical records prior to admission[cite: 4]. Default is True.
    window : tuple, optional
        The relative time window (start, end) to search if `check_current` is True,
        by default ("0h", "24h")[cite: 1, 4].
    **kwargs
        Additional keyword arguments safely absorbed by the orchestrator.

    Returns
    -------
    int
        1 if any record matches the target codes and satisfies the query expression, otherwise 0.
    """
    if check_current:
        df = ctx.get_current(domain, window=window, columns=[code_col, value_col])
    else:
        df = ctx.get_historical(domain, columns=[code_col, value_col])

    if df.empty:
        return 0

    target_df = df[match_codes(df[code_col], codes)]
    if target_df.empty:
        return 0

    target_df.loc[:, value_col] = pd.to_numeric(target_df[value_col], errors='coerce')
    match_df = target_df.query(query_expr)

    if not match_df.empty:
        return 1

    return 0


def derive_from_keyword(
        ctx: EpisodeContext,
        domain: str,
        keywords: list,
        text_col: str = "name",
        check_historical: bool = True,
        **kwargs
) -> int:
    """
    Evaluates clinical text columns for the presence of specific substring keywords.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data[cite: 4].
    domain : str
        The clinical domain table to query (e.g., 'medications', 'notes')[cite: 4].
    keywords : list
        A list of substring keywords to search for within the target text column.
        The evaluation is case-insensitive.
    text_col : str, optional
        The dataframe column containing the text to evaluate, by default "name".
    check_historical : bool, optional
        If True, evaluates strictly historical records prior to admission[cite: 4].
        If False, evaluates current records during the admission[cite: 4]. Default is True.
    **kwargs
        Additional keyword arguments safely absorbed by the orchestrator.

    Returns
    -------
    int
        1 if any record contains at least one of the specified keywords, otherwise 0.
    """
    if check_historical:
        df = ctx.get_historical(domain)
    else:
        df = ctx.get_current(domain)

    if df.empty:
        return 0

    text_series = df[text_col].fillna("").astype(str)

    pattern = '|'.join(keywords)
    if text_series.str.contains(pattern, case=False, na=False).any():
        return 1

    return 0


def derive_from_history_code(
        ctx: EpisodeContext,
        codes: list,
        res195_codes: list = None,
        target_domains: list = None,
        include_current: bool = False,
        window: tuple = ("0h", "24h"),
        **kwargs
) -> int:
    """
    Generic extractor checking target codes in historical records.
    It has the option to also include current records.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data.
    codes : list
        Primary list of target medical codes to search for.
    res195_codes : list, optional
        Additional Res195 codes to append to the primary target codes, by default None.
    target_domains : list, optional
        A list of clinical domains to search (e.g., ["problems", "diagnoses"]).
        If None, defaults to ["problems"].
    include_current : bool, optional
        If True, searches records within the current admission window in addition
        to the strictly historical records. Default is False.
    window : tuple, optional
        The time window (start, end) relative to the index admission to search
        if `include_current` is True, by default ("0h", "24h").
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int
        Returns 1 if a match for the target codes is found in the specified domains
        and temporal bounds, otherwise returns 0.
    """
    all_target_codes = list(codes)
    if res195_codes:
        all_target_codes.extend(res195_codes)

    if not all_target_codes:
        return 0

    domains_to_search = target_domains or ["problems"]

    for domain in domains_to_search:
        # 1. Always check strictly historical records first
        history_df = ctx.get_historical(domain, columns=["code"])
        if not history_df.empty and match_codes(history_df["code"], all_target_codes).any():
            return 1

        # 2. Optionally check the current admission window if the flag is enabled[cite: 9]
        if include_current:
            current_df = ctx.get_current(domain, window=window, columns=["code"])
            if not current_df.empty and match_codes(current_df["code"], all_target_codes).any():
                return 1

    return 0


import numpy as np


def derive_score_from_rules(
        ctx: EpisodeContext,
        domain: str,
        codes: list,
        rules: list,
        value_col: str = "value",
        code_col: str = "code",
        check_current: bool = True,
        window: tuple = ("0h", "24h"),
        agg_method: str = "max",
        **kwargs
) -> int:
    """
    Evaluates numeric clinical values against a set of expressions to assign points.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data.
    domain : str
        The clinical domain table to query (e.g., 'vitals', 'labs').
    codes : list
        Target clinical codes to filter the domain data.
    rules : list of dicts
        A list of dictionaries defining the scoring logic.
        Format: [{"expr": "value <= 9", "points": 4}, {"expr": "10 <= value <= 12", "points": 2}]
    value_col : str, optional
        The column containing numeric values (default: "value").
    code_col : str, optional
        The column containing clinical codes (default: "code").
    check_current : bool, optional
        If True, evaluates records within the admission window. Default is True.
    window : tuple, optional
        The time window to search, by default ("0h", "24h").
    agg_method : str, optional
        How to aggregate multiple readings ("max", "sum", "min"). Default is "max".

    Returns
    -------
    int
        The aggregated score based on the rules, or 0 if no records match.
    """
    # 1. Fetch temporal data
    if check_current:
        df = ctx.get_current(domain, window=window, columns=[code_col, value_col])
    else:
        df = ctx.get_historical(domain, columns=[code_col, value_col])

    if df.empty:
        return 0

    # 2. Filter by target codes
    target_df = df[match_codes(df[code_col], codes)].copy()
    if target_df.empty:
        return 0

    # 3. Clean and isolate numeric values
    target_df[value_col] = pd.to_numeric(target_df[value_col], errors='coerce')
    target_df = target_df.dropna(subset=[value_col])
    if target_df.empty:
        return 0

    # 4. Evaluate rules sequentially
    conditions = []
    choices = []
    for rule in rules:
        try:
            # Pandas eval executes strings like 'value <= 9' dynamically
            mask = target_df.eval(rule["expr"])
            conditions.append(mask)
            choices.append(rule.get("points", 0))
        except Exception as e:
            print(f"Warning: Failed to evaluate rule '{rule.get('expr')}': {e}")
            continue

    if not conditions:
        return 0

    # 5. Map evaluated masks to point values
    target_df['score'] = np.select(conditions, choices, default=0)

    # 6. Aggregate multiple readings (e.g., finding the worst GCS score in 24 hours)
    if agg_method == "max":
        return int(target_df['score'].max())
    elif agg_method == "sum":
        return int(target_df['score'].sum())
    elif agg_method == "min":
        return int(target_df['score'].min())

    return int(target_df['score'].max())


def derive_composite_rules(
        ctx: EpisodeContext,
        components: list,
        logic: str = "and",
        **kwargs
) -> int:
    """
    Evaluates independent sub-rules across different domains and combines them.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object.
    components : list of dicts
        A list of sub-rule configurations (specifying domain, codes, rules, etc.).
    logic : str, optional
        How to combine the results: 'and', 'or', or 'sum'. Default is 'and'.
    """
    results = []

    for comp in components:
        # Identify which utility function to run for this specific component
        extractor_type = comp.get("extractor_type", "rules")

        try:
            if extractor_type == "rules":
                val = derive_score_from_rules(ctx, **comp)
            elif extractor_type == "expression":
                val = derive_from_expression(ctx, **comp)
            elif extractor_type == "history":
                val = derive_from_history_code(ctx, **comp)
                val = val * comp.get("points", 1) # Bool to point value
            elif extractor_type == "keyword":
                val = derive_from_keyword(ctx, **comp)
                val = val * comp.get("points", 1) # Bool to point value
            else:
                val = 0

            results.append(val)
        except Exception as e:
            print(f"ERROR in extractor [{extractor_type}] with config {comp}")
            print(f"Details: {e}")
            results.append(0)

    # Combine the results based on the requested logic
    logic = str(logic).lower()
    if logic == "and":
        return 1 if all(results) and len(results) > 0 else 0
    elif logic == "or":
        return 1 if any(results) else 0
    elif logic == "sum":
        return sum(results)
    elif logic == "max":
        return max(results) if results else 0

    return 0
