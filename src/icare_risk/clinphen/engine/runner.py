"""FeatureMatrixBuilder — orchestrates the whole pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config.schema import SchemaConfig
from ..io.cohort_cache import CohortDataCache
from ..io.loaders import DuckDBSource
from ..registry.registry import DEFAULT_REGISTRY, PhenotypeRegistry, PhenotypeSpec
from ..temporal.context import EpisodeContext


def _collect_spec_domains(spec: PhenotypeSpec) -> set:
    """
    Returns the full set of clinical domains a phenotype spec actually requires:
    the top-level `spec.domains` list UNION any domain(s) declared inside nested
    `kwargs["components"]` entries (used by composite phenotypes, e.g.
    `derive_composite_rules`).

    ROOT CAUSE THIS FIXES:
    The previous domain-collection loop in `build()` only ever inspected the
    top-level `spec.domains` list:

        for spec in self.specs:
            for d in spec.domains:
                domains_needed.add(d)

    Direct phenotypes (function=derive_from_history_code, etc.) declare their
    domain(s) at the top level in YAML (`domains: [problems]`), so this worked
    fine for them. Composite phenotypes (function=derive_composite_rules)
    instead declare the domain *per component* (`kwargs["components"][i]["domain"]`),
    since different components can legitimately target different domains. A
    composite phenotype's top-level `domains:` key is therefore often left empty
    in YAML -- which is entirely reasonable authoring, but the collection loop
    above silently treated that as "this phenotype needs nothing".

    Concretely, for an ISOLATED run of just the composite phenotype
    (e.g. `build(phenotypes=["charlson_hx_hiv2"])`):
      1. `domains_needed` ends up `[]` (empty).
      2. `cache.preload([], ...)` never loads the `'problems'` table for anyone.
      3. `EpisodeContext.from_cache(..., domains=[], ...)` never puts a
         `'problems'` key into `ctx._tables` at all.
      4. Inside `has_codes()`, `df = self._tables.get('problems')` returns
         `None` -> the function returns `False`/`0` -- silently, with no
         exception, no warning.

    This is exactly why `charlson_hx_hiv` (direct) and `charlson_hx_hiv2`
    (composite) produced different results despite identical `domain`/`codes`/
    `window`/`code_col` values being logged inside `derive_from_history_code`:
    those prints only show the function's *own* arguments, which were resolved
    correctly. The failure happens one layer deeper, on data that was never
    loaded into that particular `ctx` in the first place -- invisible to that
    print. It was also easy to miss because running the composite phenotype
    *alongside* any other phenotype that happens to declare `domains: [problems]`
    at the top level "accidentally" pulls the table in anyway, masking the bug
    whenever multiple phenotypes are evaluated together.

    Fix: also scan every component dict inside `kwargs["components"]` for a
    `domain` (singular) or `domains` (plural, list) key and fold those into the
    required set, so a composite phenotype's true data dependencies are always
    fully captured regardless of which other phenotypes are in the same run.
    """
    needed = set(spec.domains)
    for comp in (spec.kwargs.get("components") or []):
        if not isinstance(comp, dict):
            continue
        comp_domain = comp.get("domain", comp.get("domains"))
        if not comp_domain:
            continue
        if isinstance(comp_domain, (list, tuple, set)):
            needed.update(comp_domain)
        else:
            needed.add(comp_domain)
    return needed

class FeatureMatrixBuilder:
    def __init__(
        self,
        schema: SchemaConfig,
        registry: PhenotypeRegistry = DEFAULT_REGISTRY,
        source: Optional[DuckDBSource] = None,
        phenotypes: Optional[List[str]] = None,
    ):
        self.schema = schema
        self.registry = registry
        self.source = source or DuckDBSource()
        self.specs: List[PhenotypeSpec] = registry.select(phenotypes)

    def build(self, subjects: Optional[List] = None,
                    episodes: Optional[pd.DataFrame] = None,
                    deduplicate_by: Optional[List[str]] = None,
                    raise_on_error: bool = True,
                    show_progress: bool = True) -> pd.DataFrame:
        """
        Build the phenotype feature matrix across all cohort episodes.

        Loads or normalizes admission episodes, preloads required clinical domain
        tables into an in-memory cohort cache, and evaluates all selected phenotypes
        for each episode.

        Parameters
        ----------
        subjects : list of str or int, optional
            List of subject identifiers to filter processing. If None, all unique
            subjects present in the episodes DataFrame are evaluated.
        episodes : pandas.DataFrame, optional
            Custom DataFrame containing admission episodes. If None, episodes are
            loaded from the configured DuckDB source.
        deduplicate_by : str or list of str, optional
            Logical (e.g., 'spell', 'encntr') or physical column name(s) used to drop
            duplicate episode rows prior to feature computation.
        raise_on_error : bool, default=False
            If True, halts execution on the first unhandled phenotype failure. If
            False, captures the exception string inside output failure columns.
        show_progress : bool, default=False
            If True, renders an interactive `tqdm` progress bar during episode
            iteration.

        Returns
        -------
        pandas.DataFrame
            Matrix indexed by episode identifier columns (`spell`, `encntr`) containing
            original episode metadata alongside computed phenotype features.

        Raises
        ------
        ValueError
            If a registered phenotype requires a domain table that is absent from
            `schema_config.yaml`.

        Notes
        -----
        Domain data preloading is executed in batch per unique subject before
        processing individual episodes to ensure O(1) in-memory lookups.
        """
        # 1. Load raw episodes or use custom DataFrame
        if episodes is not None:
            raw_episodes = episodes
        else:
            raw_episodes = self.source.load_episodes(self.schema.episodes, subjects=subjects)

        if raw_episodes.empty:
            return pd.DataFrame()

        # 2. Normalize columns to logical names ("subject", "spell", "index_admission", etc.)
        df_episodes = self.schema.episodes.normalize_frame(raw_episodes)

        if subjects is not None and "subject" in df_episodes.columns:
            df_episodes = df_episodes[df_episodes["subject"].isin(subjects)]

        # 3. Handle deduplication by physical name (e.g. 'SPELL_IDENTIFIER') or logical name (e.g. 'spell')
        if deduplicate_by is not None:
            if isinstance(deduplicate_by, str):
                deduplicate_by = [deduplicate_by]

            # Map physical deduplication names to normalized logical column names
            alias_map = {
                self.schema.episodes.spell: "spell",
                self.schema.episodes.encounter: "encntr",
                self.schema.episodes.subject: "subject",
            }

            resolved_dedup_cols = []
            for col in deduplicate_by:
                logical_col = alias_map.get(col, col)
                if logical_col in df_episodes.columns:
                    resolved_dedup_cols.append(logical_col)
                elif col in df_episodes.columns:
                    resolved_dedup_cols.append(col)

            if resolved_dedup_cols:
                df_episodes = df_episodes.drop_duplicates(subset=resolved_dedup_cols, keep="first")

        available_domains = set(self.schema.domains.keys())
        domains_needed = set()

        for spec in self.specs:
            # FIX: use _collect_spec_domains() instead of iterating spec.domains
            # directly, so composite phenotypes' nested component domains are
            # captured too. See _collect_spec_domains() docstring above for the
            # full root-cause explanation.

            #for d in spec.domains:
            spec_domains = _collect_spec_domains(spec)
            for d in spec_domains:
                if d not in available_domains:
                    raise ValueError(
                        f"Configuration Error: Phenotype '{spec.name}' requires domain '{d}'"
                    )
                domains_needed.add(d)

        domains_needed = sorted(domains_needed)

        import time
        t0 = time.perf_counter()

        # 4. Preload cache using normalized 'subject' column
        cache = CohortDataCache(self.schema, source=self.source).preload(
            domains_needed, subjects=df_episodes["subject"].unique().tolist()
        )

        # 5. Fast dictionary loop with optional progress bar
        episodes_records = df_episodes.to_dict(orient="records")

        # Setup the iterator based on show_progress
        iterator = episodes_records
        if show_progress:
            print(f"Loaded clinical domains in {time.perf_counter() - t0:.2f}s")
            try:
                # Use tqdm.auto for seamless Jupyter Notebook rendering
                from tqdm.auto import tqdm
                iterator = tqdm(episodes_records, desc="Processing Episodes", total=len(episodes_records))
            except ImportError:
                import logging
                logging.warning(
                    "The 'tqdm' library is not installed. Progress bar disabled. Run 'pip install tqdm' to enable.")

        rows: List[Dict] = []
        for ep in iterator:
            ctx = EpisodeContext.from_cache(
                cache,
                subject=ep["subject"],
                index_admission=ep["index_admission"],
                domains=domains_needed,
                index_discharge=ep.get("index_discharge"),
            )
            row: Dict = ep

            for spec in self.specs:
                self._run_one(spec, ctx, row, raise_on_error)
            rows.append(row)

        df = pd.DataFrame(rows)

        index_cols = [col for col in ["spell", "encntr"] if col in df.columns]
        return df.set_index(index_cols) if index_cols else df

    @staticmethod
    def _run_one(spec: PhenotypeSpec,
                 ctx: EpisodeContext,
                 row: Dict,
                 raise_on_error: bool) -> None:
        try:
            kwargs = getattr(spec, "kwargs", {})
            result = spec.func(ctx, domains=spec.domains, **kwargs)
        except Exception as exc:  # noqa: BLE001 - isolate per-phenotype failures
            if raise_on_error:
                raise
            row[spec.name] = None
            row[f"{spec.name}__error"] = f"{type(exc).__name__}: {exc}"
            return

        if isinstance(result, dict):
            for key, value in result.items():
                row[f"{spec.name}__{key}"] = value
        else:
            row[spec.name] = result
