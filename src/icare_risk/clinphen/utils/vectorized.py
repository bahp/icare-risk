# Libraries
import pandas as pd

from icare_risk.clinphen.utils.filtering import match_codes
from typing import Dict, List, Set, Any, Optional, Union, Tuple

# ------------------------------------------------------------------
# Helper methods
# ------------------------------------------------------------------
def _format_td(td: pd.Timedelta) -> str:
    """Human-readable rendering of a resolved Timedelta bound for verbose logging."""
    if td == pd.Timedelta.min:
        return "-inf"
    if td == pd.Timedelta(0):
        return "0h"
    return f"{td.total_seconds() / 3600:+.1f}h"

def _first_not_none(*values, default: Any = None) -> Any:
    """
    Returns the first value in `values` that is not None, respecting
    legitimate falsy values (e.g. False, "", 0). Falls back to `default`
    if every supplied value is None.

    This implements the resolution priority used throughout this module:
        most-specific (e.g. per-source) > YAML (phenotype config) >
        domain registry default > hardcoded fallback
    """
    for v in values:
        if v is not None:
            return v
    return default

def _default_code_extractor(config: dict, all_configs: dict) -> List[str]:
    """Default method to extract codes."""
    if "codes" in config:
        return config["codes"]
    if "keywords" in config:
        return config["keywords"]

    kwargs = config.get("kwargs", {})
    if "codes" in kwargs:
        return kwargs["codes"]
    if "keywords" in kwargs:
        return kwargs["keywords"]

    components = kwargs.get("components", [])
    extracted = []
    for comp in components:
        extracted.extend(comp.get("codes", []))
        extracted.extend(comp.get("keywords", []))

    return list(set(extracted))




def parse_phenotypes_to_batches(yaml_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses a complex clinical phenotype YAML configuration and flattens it
    into optimized execution batches grouped by domain, target column, and match type.
    """
    # Use a dictionary to group identical (domain, target_col, match_type) combinations
    grouped_batches = {}

    for phenotype_name, pheno_data in yaml_config.items():
        kwargs = pheno_data.get("kwargs", {})
        components = kwargs.get("components", [])
        parent_logic = kwargs.get("logic", "or") # Will be used in Step 2
        parent_window = kwargs.get("window")     # Phenotype-level default window

        for i, comp in enumerate(components):
            domain = comp.get("domain")
            extractor_type = comp.get("extractor_type")

            if not domain or not extractor_type:
                continue

            # NOTE: match_type / text_col / case_sensitive are intentionally
            # left as None when absent from the YAML (no hardcoded defaults
            # here). This lets `run_vectorized` tell "not specified in YAML"
            # apart from "explicitly specified", so it can correctly fall
            # back to the domain registry default only when truly needed.

            # Route 1: Keyword/Text Matching
            if extractor_type == "keyword":
                match_type = comp.get("match_type")
                target_col = comp.get("text_col")
                items = comp.get("keywords", [])
                case_sensitive = comp.get("case_sensitive")

            # Route 2: History/Code Matching
            elif extractor_type == "history":
                match_type = comp.get("match_type")
                target_col = comp.get("text_col")
                items = comp.get("codes", [])
                case_sensitive = comp.get("case_sensitive")

            else:
                continue  # Skip unknown extractors

            if not items:
                continue

            # Define the grouping key
            group_key = (domain, target_col, match_type, case_sensitive)

            if group_key not in grouped_batches:
                grouped_batches[group_key] = {
                    "domain": domain,
                    "target_col": target_col,
                    "match_type": match_type,
                    "case_sensitive": case_sensitive,
                    "configs": {}
                }

            # Create a unique sub-event name for this specific component
            sub_event_name = f"{phenotype_name}__comp{i}"

            # Resolve this component's window with the same priority chain
            # used for match_type/text_col/case_sensitive:
            #   component-level "window"  >  phenotype-level kwargs.window
            # (A domain-level default, if any, is applied later downstream
            # in `evaluate_temporal_windows`, since that's where the domain
            # registry is actually available.)
            resolved_window = _first_not_none(comp.get("window"), parent_window)

            # Standardize output: Always map target terms to "codes" for the builder
            grouped_batches[group_key]["configs"][sub_event_name] = {
                "codes": items,
                "window": resolved_window,
                "parent_phenotype": phenotype_name,
                "logic": parent_logic
            }

    return list(grouped_batches.values())

def parse_window(wspec: Any) -> Tuple[pd.Timedelta, pd.Timedelta]:
    """
    Parses flexible window configurations into explicit (min, max) pd.Timedelta
    bounds, used downstream as:  min_td <= (event_time - admission_time) < max_td

    Supported inputs
    ----------------
    None
        -> (-inf, 0h). Full lifetime lookback up to (not including) admission.

    A single bare value with NO list/tuple wrapper, e.g. 720, "720h", "-30d",
    "+7h", or a pd.Timedelta
        -> the SIGN is respected (it is never discarded/abs'd):
             negative value  -> backward window ending at admission: (value, 0h)
             positive value  -> forward window starting at admission: (0h, value)
             zero            -> degenerate (0h, 0h)
           e.g. "-720h"/-720 -> (-720h, 0h)   (lookback, as in "90 days prior")
                "7h"/+7/7    -> (0h, 7h)      (look-ahead, as in "within 7h after")
           Bare numbers are interpreted as **hours**. This mirrors the literal,
           sign-preserving behavior of the 2-element list/tuple case below, so
           a bare scalar is just shorthand for a window whose "near" bound is
           implicitly 0h (admission time).

    A 2-element list/tuple (lower, upper), e.g. ["-250h", "0h"] or [10, 24]
        -> taken LITERALLY. No sign is ever flipped here:
             ["-250h", "0h"]  -> (-250h, 0h)   # lookback window
             [10, 24]         -> (10h,  24h)    # forward-only window
             [-10, 24]        -> (-10h, 24h)    # straddling window
           Bare ints/floats are interpreted as **hours**; strings are parsed
           by pandas (`pd.Timedelta`), so "2d", "90d", "-30m", etc. all work.
           `None` in either slot falls back to -inf (lower) / 0h (upper).

    A dict wrapping any of the above under "window" / "kwargs.window" /
    "kwargs.components[*].window" (first match wins) -- used when a raw
    phenotype config dict is passed in directly instead of an already
    extracted window spec.
    """
    LIFETIME_MIN = pd.Timedelta.min
    ZERO = pd.Timedelta(0)

    def _to_bound(val, default):
        """Resolve a single list/tuple endpoint to a concrete Timedelta."""
        if val is None:
            return default
        if isinstance(val, pd.Timedelta):
            return val
        if isinstance(val, bool):
            # bool is a subclass of int -- guard against True/False leaking in
            return default
        if isinstance(val, (int, float)):
            # FIX: previously `pd.Timedelta(val)` on a bare number defaults to
            # NANOSECONDS, so e.g. [10, 24] silently collapsed to ~0 instead
            # of (10h, 24h). Bare numeric bounds are now explicit hours.
            return pd.Timedelta(hours=val)
        if isinstance(val, str):
            return pd.Timedelta(val)
        return default

    def _scalar_to_window(td: pd.Timedelta) -> Tuple[pd.Timedelta, pd.Timedelta]:
        """
        Turns a single resolved Timedelta into a (min, max) window, preserving
        its sign instead of discarding it (FIX: previously every bare scalar
        was forced into a backward-only window via abs(), so a positive,
        unsigned value like "7h" was silently treated identically to "-7h"
        -- both collapsed to (-7h, 0h). Now:
            negative -> (td, 0h)   backward/lookback window ending at admission
            positive -> (0h, td)   forward/look-ahead window starting at admission
            zero     -> (0h, 0h)
        """
        if td < ZERO:
            return td, ZERO
        return ZERO, td

    if wspec is None:
        return LIFETIME_MIN, ZERO

    if isinstance(wspec, dict):
        window_val = None

        if "window" in wspec:
            window_val = wspec["window"]
        elif "kwargs" in wspec and "window" in wspec["kwargs"]:
            window_val = wspec["kwargs"]["window"]
        elif "kwargs" in wspec and "components" in wspec["kwargs"]:
            # Check within the components list and grab the first defined window
            for comp in wspec["kwargs"]["components"]:
                if isinstance(comp, dict) and "window" in comp:
                    window_val = comp["window"]
                    break

        # Override spec with the found window value (or None)
        wspec = window_val

    if wspec is None:
        return LIFETIME_MIN, ZERO

    if isinstance(wspec, (list, tuple)) and len(wspec) == 2:
        lower, upper = wspec[0], wspec[1]
        min_td = _to_bound(lower, LIFETIME_MIN)
        max_td = _to_bound(upper, ZERO)
        return min_td, max_td

    # FIX: a bare scalar (no list/tuple) previously only handled `str` and
    # silently fell through to the hardcoded lifetime default for any other
    # type (e.g. a plain int like `window: 720`), discarding the user's
    # intent entirely. Now int/float/Timedelta scalars are treated the same
    # way as the string case: a lookback magnitude ending at admission (0h).
    if isinstance(wspec, pd.Timedelta):
        return _scalar_to_window(wspec)

    if isinstance(wspec, (int, float)) and not isinstance(wspec, bool):
        return _scalar_to_window(pd.Timedelta(hours=wspec))

    if isinstance(wspec, str):
        return _scalar_to_window(pd.Timedelta(wspec))

    return LIFETIME_MIN, ZERO

# ------------------------------------------------------------------
# Main functions
# ------------------------------------------------------------------
def build_historical_events_table(
    df: pd.DataFrame,
    configs: Dict[str, dict],
    subject_col: str = "SUBJECT",
    time_col: str = "PROBLEM_DT_TM",
    target_col: str = "PROBLEM_CODE",
    first_occurrence_only: bool = True,
    match_type: str = "exact",
    case_sensitive: bool = False,
    strip_dots: bool = True,
    extract_codes_fn: Optional[callable] = None,
) -> pd.DataFrame:
    """"""
    if df.empty or not configs:
        out_cols = (
            [subject_col, "event_name", "first_occurrence_date"]
            if first_occurrence_only
            else list(df.columns) + ["event_name"]
        )
        return pd.DataFrame(columns=out_cols)

    extractor = extract_codes_fn or _default_code_extractor
    matched_records = []

    for event_name, config in configs.items():
        codes = extractor(config, configs)
        if not codes:
            continue

        curr_target_col = config.get("target_col", target_col)
        curr_match_type = config.get("match_type", match_type)
        curr_case_sensitive = config.get("case_sensitive", case_sensitive)
        curr_strip_dots = config.get("strip_dots", strip_dots)

        if curr_target_col not in df.columns:
            continue

        series = df[curr_target_col]
        is_match = match_codes(
            series=series,
            target_codes=codes,
            match_type=curr_match_type,
            case_sensitive=curr_case_sensitive,
            strip_dots=curr_strip_dots,
        )

        if is_match.any():
            matched_df = df[is_match].copy()
            matched_df["event_name"] = event_name
            matched_records.append(matched_df)

    if not matched_records:
        out_cols = (
            [subject_col, "event_name", "first_occurrence_date"]
            if first_occurrence_only
            else list(df.columns) + ["event_name"]
        )
        return pd.DataFrame(columns=out_cols)

    combined_matches = pd.concat(matched_records, ignore_index=True)

    if first_occurrence_only:
        first_occ = (
            combined_matches.groupby([subject_col, "event_name"])[time_col]
            .min()
            .reset_index()
        )
        first_occ.rename(
            columns={time_col: "first_occurrence_date"}, inplace=True
        )
        return first_occ

    return combined_matches


def evaluate_temporal_windows(
    episodes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_configs: Optional[Dict[str, Any]] = None,
    domain_default_windows: Optional[Dict[str, Any]] = None,
    subject_col: str = "subject",
    admission_col: str = "index_admission",
    event_col: str = "event_name",
    event_time_col: str = "date",
    domain_col: str = "domain",
    prefix: str = "hx_",
    default_window: Any = (None, "0h"),
    verbose: int = 1
) -> pd.DataFrame:
    """Evaluates temporal phenotype windows per episode directly from long event logs.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        Cohort DataFrame with patient episodes and admission dates.
    events_df : pd.DataFrame
        Long event log containing [subject_col, event_col, event_time_col],
        and optionally `domain_col` (as produced by `run_vectorized`).
    window_configs : dict, optional
        Mapping of event_name to window specs (tuples, strings, or loaded YAML dicts).
        This is the most specific / highest-priority source of truth.
    domain_default_windows : dict, optional
        Mapping of domain name -> window spec, used as a fallback default for
        any event whose name is not present in `window_configs`. Mirrors the
        same domain-registry-default pattern used for match_type/text_col
        elsewhere in this module.
    subject_col, admission_col, event_col, event_time_col, domain_col : str
        Column names for indexing and temporal matching.
    prefix : str
        Prefix added to output binary indicator columns.
    default_window : tuple or string
        Final hardcoded fallback applied when neither `window_configs` nor
        `domain_default_windows` has an entry for an event.
    verbose : int
        When > 0, prints, per event_name: which window spec was resolved,
        which source it came from (event-specific config / domain default /
        hardcoded fallback), the parsed (min, max) bounds, and how many
        candidate events fell inside vs. outside that window. This is the
        actual point in the pipeline where window resolution + filtering
        happens, so it is the most accurate place to surface this info
        (as opposed to `run_vectorized`, which only sees the YAML-resolved
        window before any domain-default fallback is applied).

    Resolution priority for each event's window
    --------------------------------------------
        window_configs[event_name]  >  domain_default_windows[domain]  >  default_window

    Returns
    -------
    pd.DataFrame
        Copy of `episodes_df` with binary indicator columns for matched phenotypes.
    """
    if events_df.empty or episodes_df.empty:
        return episodes_df.copy()

    window_configs = window_configs or {}
    domain_default_windows = domain_default_windows or {}

    episodes = episodes_df.copy()
    episodes["_episode_id"] = episodes.index

    merge_cols = [subject_col, event_col, event_time_col]
    has_domain = domain_col in events_df.columns
    if has_domain:
        merge_cols.append(domain_col)

    # 1. Join event stream to cohort
    merged = episodes[["_episode_id", subject_col, admission_col]].merge(
        events_df[merge_cols],
        on=subject_col,
        how="inner"
    )

    if merged.empty:
        return episodes.drop(columns=["_episode_id"])

    # 2. Compute relative time delta
    merged["time_delta"] = pd.to_datetime(merged[event_time_col]) - pd.to_datetime(merged[admission_col])

    if verbose > 0:
        print("\n--- Evaluating Temporal Windows ---")
        print(f"Events to evaluate: {len(merged):,} across {merged[event_col].nunique()} event name(s).\n")

    # 3. Apply per-event window filtering
    valid_mask = pd.Series(False, index=merged.index)
    window_debug_rows = []

    for event_name, group in merged.groupby(event_col):

        # Priority: explicit per-event config > domain-level default > hardcoded fallback
        event_specific = window_configs.get(event_name)
        domain_default = None
        domain_label = None
        if has_domain:
            domain_vals = group[domain_col].dropna().unique()
            if len(domain_vals) > 0:
                domain_label = domain_vals[0]
                domain_default = domain_default_windows.get(domain_label)

        if event_specific is not None:
            source_used = "event_config"
        elif domain_default is not None:
            source_used = f"domain_default[{domain_label}]"
        else:
            source_used = "hardcoded_default"

        spec = _first_not_none(
            event_specific,
            domain_default,
            default_window
        )
        min_td, max_td = parse_window(spec)

        event_mask = (group["time_delta"] >= min_td) & (group["time_delta"] < max_td)
        valid_mask.loc[group.index] = event_mask

        if verbose > 0:
            n_candidates = len(group)
            n_valid = int(event_mask.sum())
            window_debug_rows.append({
                "event_name": event_name,
                "domain": domain_label,
                "window_source": source_used,
                "window_spec": spec,
                "resolved_window": f"[{_format_td(min_td)}, {_format_td(max_td)})",
                "candidates": n_candidates,
                "within_window": n_valid,
                "dropped": n_candidates - n_valid,
            })

    if verbose > 0 and window_debug_rows:
        debug_df = pd.DataFrame(window_debug_rows)
        print(debug_df.to_string(index=False))
        print(
            f"\nTotal: {debug_df['candidates'].sum():,} candidate event(s), "
            f"{debug_df['within_window'].sum():,} fell within their resolved window, "
            f"{debug_df['dropped'].sum():,} dropped as out-of-window.\n"
        )

    valid_events = merged[valid_mask]

    if valid_events.empty:
        if verbose > 0:
            print("[!] No events remained after window filtering.")
        return episodes.drop(columns=["_episode_id"])

    # 4. Unstack into binary indicator columns
    indicators = (
        valid_events.groupby(["_episode_id", event_col])
        .size()
        .unstack(fill_value=0)
        .astype(bool)
        .astype(int)
    )

    indicators.columns = [f"{prefix}{col}" for col in indicators.columns]

    # 5. Join back to full cohort
    result = episodes \
        .merge(indicators, on="_episode_id", how="left") \
        .drop(columns=["_episode_id"])
    result[indicators.columns] = result[indicators.columns].fillna(0).astype(int)

    return result


def aggregate_composite_phenotypes(
    df: pd.DataFrame,
    phenotype_configs: Dict[str, Any],
    prefix: str = "hx_",
    drop_components: bool = False,
) -> pd.DataFrame:

    result = df.copy()
    all_component_cols_to_drop: List[str] = []

    for phenotype_name, config in phenotype_configs.items():
        kwargs = config.get("kwargs", {})
        logic = str(kwargs.get("logic", "or")).lower()
        components = kwargs.get("components", [])

        component_cols: List[str] = []

        # 1. Try resolving via explicit config keys / fallback naming
        for idx, comp in enumerate(components):
            comp_id = (
                comp.get("name")
                or comp.get("component_id")
                or comp.get("event_name")
                or f"{phenotype_name}__comp{idx}"
            )

            prefixed_id = (
                f"{prefix}{comp_id}"
                if not comp_id.startswith(prefix)
                else comp_id
            )

            if prefixed_id in result.columns:
                component_cols.append(prefixed_id)
            elif comp_id in result.columns:
                component_cols.append(comp_id)

        # 2. Fully generic fallback: dynamically match any columns associated
        #    with this phenotype's components
        if not component_cols:
            clean_name = (
                phenotype_name[len(prefix):]
                if phenotype_name.startswith(prefix)
                else phenotype_name
            )
            component_cols = [
                c for c in result.columns
                if clean_name in c and "__comp" in c
            ]

        parent_col = (
            f"{prefix}{phenotype_name}"
            if not phenotype_name.startswith(prefix)
            else phenotype_name
        )

        if not component_cols:
            result[parent_col] = 0
            continue

        # Keep track of component columns for optional cleanup
        all_component_cols_to_drop.extend(component_cols)

        # 3. Apply aggregation logic
        if logic in ("or", "any", "max"):
            result[parent_col] = (result[component_cols].max(axis=1) > 0).astype(int)
        elif logic in ("and", "all", "min"):
            result[parent_col] = (result[component_cols].min(axis=1) > 0).astype(int)
        elif logic == "sum":
            result[parent_col] = result[component_cols].sum(axis=1).astype(int)
        else:
            result[parent_col] = (result[component_cols].max(axis=1) > 0).astype(int)

    # 4. Clean up component columns if requested
    if drop_components and all_component_cols_to_drop:
        unique_comps_to_drop = list(set(all_component_cols_to_drop))
        result = result.drop(columns=[c for c in unique_comps_to_drop if c in result.columns])

    return result


def run_vectorized(domains, phenotypes, verbose: int = 1):
    """
    Executes vectorized phenotype extraction across registered domains,
    supporting multi-source target columns (e.g., SNOMED vs ICD-10) with
    individual match types.

    Resolution priority for target_col / match_type / case_sensitive:
        1. Per-source override (registry's `sources` entry), when present
        2. YAML phenotype config (component's text_col / match_type / case_sensitive)
        3. Domain registry default (registry's top-level target_col / match_type / case_sensitive)
        4. Hardcoded fallback ("code" / "exact" / False)
    """

    # Libraries
    from icare_risk.clinphen.utils.vectorized import (
        parse_phenotypes_to_batches,
        build_historical_events_table
    )

    if verbose > 0:
        print("\n--- Running Vectorized Bench ---\n")
        print("--- 1. Parsing Phenotypes YAML ---")

    batches = parse_phenotypes_to_batches(phenotypes)

    if verbose > 0:
        print(f"Generated {len(batches)} execution batch(es).\n")
        print("--- 2. Executing Batches ---")

    # 2. Process each batch
    all_extracted_events = []
    for b_idx, batch in enumerate(batches, 1):
        domain_name = batch["domain"]

        if domain_name not in domains:
            print(f"Warning: Domain '{domain_name}' not registered. Skipping.")
            continue

        registry_info = domains[domain_name]
        df_target = registry_info["df"]
        time_col = registry_info["time_col"]
        subject_col = registry_info["subject_col"]

        sources = registry_info.get("sources")
        if not sources:

            # YAML (batch) value takes priority over the registry's.
            t_col = _first_not_none(
                batch.get("target_col"),
                registry_info.get("target_col"),
                default="code",
            )
            match_type = _first_not_none(
                batch.get("match_type"),
                registry_info.get("match_type"),
                default="exact",
            )
            case_sensitive = _first_not_none(
                batch.get("case_sensitive"),
                registry_info.get("case_sensitive"),
                default=False,
            )
            sources = [{
                "target_col": t_col,
                "match_type": match_type,
                "case_sensitive": case_sensitive,
            }]

        # Loop through each source defined for this domain
        for s_idx, source in enumerate(sources):
            t_col = source["target_col"]
            match_type = _first_not_none(
                source.get("match_type"),
                batch.get("match_type"),
                registry_info.get("match_type"),
                default="exact",
            )
            case_sensitive = _first_not_none(
                source.get("case_sensitive"),
                batch.get("case_sensitive"),
                registry_info.get("case_sensitive"),
                default=False,
            )

            # ---------------------------------------------------------
            # DEBUG SNIPPET
            # ---------------------------------------------------------
            if verbose > 0:
                print(f"\n--- DEBUG BATCH {b_idx} (Source {s_idx+1}) [{domain_name} -> {t_col}] ---")
                print(f"Match Type: {match_type}, Case Sensitive: {case_sensitive}")

                # Extract search values (handles both dicts and raw strings/lists)
                search_values = []
                for c in batch["configs"]:
                    if isinstance(c, dict):
                        val = c.get("value") or c.get("code") or c.get("values") or c
                    else:
                        val = c

                    if isinstance(val, list):
                        search_values.extend(val)
                    else:
                        search_values.append(val)

                print("YAML is looking for (first 5):")
                for val in search_values[:5]:
                    print(f"  - {val}")

                if t_col in df_target.columns:
                    top_common = df_target[t_col].value_counts().head(5)
                    print("DataFrame top 5 most common values (with counts):")
                    if not top_common.empty:
                        for val, count in top_common.items():
                            print(f"  - {val}: {count:,} occurrence(s)")
                    else:
                        print("  - Column is empty or contains only nulls.")
                else:
                    print(f"CRITICAL ERROR: Column '{t_col}' does not exist in DataFrame!")
                    continue
            # ---------------------------------------------------------

            # Run historical event extraction for this specific source
            res = build_historical_events_table(
                df=df_target,
                configs=batch["configs"],
                subject_col=subject_col,
                time_col=time_col,
                target_col=t_col,
                first_occurrence_only=False,       # Keep all for windowing
                match_type=match_type,
                case_sensitive=case_sensitive
            )

            if not res.empty:
                # Preserve the event timestamp under a unified name for Step 2
                res.rename(columns={time_col: "event_dt_tm"}, inplace=True)
                res['domain'] = domain_name
                all_extracted_events.append(res)
                print(f"-> Matched {len(res)} record(s) using column '{t_col}' with match_type='{match_type}'.")

    if not all_extracted_events:
        print("\n[!] No matching records found across any domain/source.")
        return pd.DataFrame()

    combined_df = pd.concat(all_extracted_events, ignore_index=True)

    if verbose > 0:
        print("\n--- 3. Extracted Event Timeline (With Component Windows) ---")
        output_cols = [
            "SUBJECT",
            "domain",
            "event_name",
            "event_dt_tm",
        ]
        valid_output_cols = [c for c in output_cols if c in combined_df.columns]
        print(combined_df[valid_output_cols].head(10).to_string(index=False))
    print(f"Extracted {len(combined_df):,} total matching events across all domains.")

    return combined_df

