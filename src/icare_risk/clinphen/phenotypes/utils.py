# Libraries
import operator
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict, Callable

from icare_risk.clinphen.utils.filtering import match_codes
from icare_risk.clinphen.temporal.context import EpisodeContext


def _fetch_domain_data(
        ctx: EpisodeContext,
        domains: Union[str, List[str]],
        window: Optional[Tuple[Optional[str], Optional[str]]],
        columns: list
) -> pd.DataFrame:
    """
    Helper to fetch, validate, and combine data across multiple domains.
    Reports if a domain does not exist in the context or is missing required columns.
    """
    # Polymorphic normalization at the boundary
    if isinstance(domains, str):
        domain_list = [domains]
    elif isinstance(domains, (list, tuple)):
        domain_list = list(domains)
    else:
        return pd.DataFrame(columns=columns)

    dfs = []
    for dom in domain_list:
        # Report if domain does not exist in the EpisodeContext
        if dom not in ctx._tables:
            print(f"WARNING: Domain '{dom}' does not exist in the current context.")
            continue

        df = ctx.get_window(dom, window=window, columns=columns)
        if df.empty:
            continue

        # Report if required columns are missing in the dataframe
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            print(f"WARNING: Required columns {missing_cols} not found in domain '{dom}'.")
            continue

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=columns)

    return pd.concat(dfs, ignore_index=True)



def derive_from_expression(
        ctx: EpisodeContext,
        domains: str,
        codes: list,
        query_expr: str,
        value_col: str = "value",
        code_col: str = "code",
        window: Optional[Tuple[Optional[str], Optional[str]]] = ("0h", "24h"),
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
        The temporal context object containing the patient's episode data.
    codes : list
        A list of target clinical codes to filter the domain data before evaluating the expression.
    query_expr : str
        A valid pandas query expression string referencing the `value_col` (e.g., 'value < 4.0').
    domains : str
        The clinical domain table to query (e.g., 'labs', 'vitals').
    value_col : str, optional
        The dataframe column containing the numeric values to evaluate, by default "value".
    code_col : str, optional
        The dataframe column containing the clinical codes to match against, by default "code".
    window : tuple, optional
        The relative time window (start, end) to search.
    **kwargs
        Additional keyword arguments safely absorbed by the orchestrator.

    Returns
    -------
    int
        1 if any record matches the target codes and satisfies the query expression, otherwise 0.
    """
    if not codes or not domains:
        return 0

    df = _fetch_domain_data(ctx, domains, window=window, columns=[code_col, value_col])
    if df.empty:
        return 0

    target_df = df[match_codes(df[code_col], codes)].copy()
    if target_df.empty:
        return 0

    target_df[value_col] = pd.to_numeric(target_df[value_col], errors='coerce')
    match_df = target_df.query(query_expr)

    return 1 if not match_df.empty else 0
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
    """


def derive_from_keyword(
        ctx: EpisodeContext,
        domains: list,
        keywords: list,
        text_col: str = "name",
        window: Optional[Tuple[str, str]] = (None, '0h'),
        **kwargs
) -> int:
    """
    Evaluates a single clinical text column for the presence of specific keywords.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data
    keywords : list
        List of substring keywords to search for (case-insensitive).
    domains : str
        The clinical domain table to query (e.g., 'medications', 'notes')
    text_col : str, default="name"
        The specific dataframe column to evaluate.
    window : tuple, optional
        Relative time window tuple (start, end) relative to index admission (e.g., ("-2160h", "0h")).
        If None, evaluates unbounded historical records prior to admission[cite: 9].
    **kwargs
        Additional keyword arguments safely absorbed by the orchestrator.

    Returns
    -------
    int
        1 if any record contains at least one of the specified keywords, otherwise 0.
    """
    """Searches clinical text columns for substring keywords."""
    if not keywords or not domains:
        return 0

    df = _fetch_domain_data(ctx, domains, window=window, columns=[text_col])
    if df.empty:
        return 0

    text_series = df[text_col].fillna("").astype(str)
    pattern = '|'.join(keywords)

    return 1 if text_series.str.contains(pattern, case=False, na=False).any() else 0
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
    """
    if not keywords:
        return 0

    # 1. Fetch data based on domain and window
    df = ctx.get_window(domain, window=window)
    if df.empty:
        return 0

    # 2. Check for missing column and show a warning
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found in '{domain}'. Available columns: {list(df.columns)}")
        return 0

    # 2. Case-insensitive substring search
    text_series = df[text_col].fillna("").astype(str)
    pattern = '|'.join(keywords)

    if text_series.str.contains(pattern, case=False, na=False).any():
        return 1

    return 0



def derive_from_history_code(
        ctx: EpisodeContext,
        domains: Union[str, List[str]],
        codes: list,
        code_col: str = "code",
        window: Optional[Tuple[str, str]] = (None, '0h'),
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
    domains : list, optional
        A list of clinical domains to search (e.g., ["problems", "diagnoses"]).
    code_col: str, optional
        The code to check the values
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
    if not codes or not domains:
        return 0

    df = _fetch_domain_data(ctx, domains, window=window, columns=[code_col])
    if df.empty:
        return 0

    return 1 if match_codes(df[code_col], codes).any() else 0

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
    """



def derive_score_from_rules(
        ctx: EpisodeContext,
        domains: str,
        codes: list,
        rules: list,
        value_col: str = "value",
        code_col: str = "code",
        window: Optional[Tuple[Optional[str], Optional[str]]] = ("0h", "24h"),
        agg_method: str = "max",
        **kwargs
) -> int:
    """
    Evaluates numeric clinical values against a set of expressions to assign points.

    Parameters
    ----------
    ctx : EpisodeContext
        The temporal context object containing the patient's episode data.
    codes : list
        Target clinical codes to filter the domain data.
    rules : list of dicts
        A list of dictionaries defining the scoring logic.
        Format: [{"expr": "value <= 9", "points": 4}, {"expr": "10 <= value <= 12", "points": 2}]
    domains : str
        The clinical domain table to query (e.g., 'vitals', 'labs').
    value_col : str, optional
        The column containing numeric values (default: "value").
    code_col : str, optional
        The column containing clinical codes (default: "code").
    window : tuple, optional
        The time window to search, by default ("0h", "24h").
    agg_method : str, optional
        How to aggregate multiple readings ("max", "sum", "min"). Default is "max".

    Returns
    -------
    int
        The aggregated score based on the rules, or 0 if no records match.
    """
    """Evaluates numeric rules to calculate component point scores."""
    if not codes or not rules or not domains:
        return 0

    df = _fetch_domain_data(ctx, domains, window=window, columns=[code_col, value_col])
    if df.empty:
        return 0

    target_df = df[match_codes(df[code_col], codes)].copy()
    if target_df.empty:
        return 0

    target_df[value_col] = pd.to_numeric(target_df[value_col], errors='coerce')
    target_df = target_df.dropna(subset=[value_col])
    if target_df.empty:
        return 0

    conditions, choices = [], []
    for rule in rules:
        try:
            mask = target_df.eval(rule["expr"])
            conditions.append(mask)
            choices.append(rule.get("points", 0))
        except Exception:
            continue

    if not conditions:
        return 0

    target_df['score'] = np.select(conditions, choices, default=0)

    if agg_method == "sum":
        return int(target_df['score'].sum())
    elif agg_method == "min":
        return int(target_df['score'].min())

    return int(target_df['score'].max())


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

# 1. Extractor Registry (Open for extension, closed for modification)
EXTRACTOR_REGISTRY: Dict[str, Callable] = {
    "rules": derive_score_from_rules,
    "expression": derive_from_expression,
    "history": derive_from_history_code,
    "keyword": derive_from_keyword,
}

def derive_composite_rules(
        ctx: EpisodeContext,
        #domains: Optional[Union[str, List[str]]] = None,
        components: Optional[List[dict]] = None,
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
    valid_extractors = {"rules", "expression", "history", "keyword"}

    for comp in components:
        # Identify which utility function to run for this specific component
        extractor_type = comp.get("extractor_type")

        # Explicit warning if a YAML rule is mapped to an unknown type
        if extractor_type not in valid_extractors:
            print(f"""WARNING: Unknown extractor_type '{extractor_type}' found 
                   in config: {comp}. Expected one of {valid_extractors}.""")
            results.append(0)
            continue

        # 1. Resolve domain(s): checks singular 'domain', plural 'domains', or falls back to top-level 'domains'
        domains = comp.pop("domain", None)

        try:
            if extractor_type == "rules":
                val = derive_score_from_rules(ctx, domains=domains, **comp)
            elif extractor_type == "expression":
                val = derive_from_expression(ctx, domains=domains, **comp)
            elif extractor_type == "history":
                val = derive_from_history_code(ctx, domains=domains, **comp)
                val = val * comp.get("points", 1) # Bool to point value
            elif extractor_type == "keyword":
                val = derive_from_keyword(ctx, domains=domains, **comp)
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


