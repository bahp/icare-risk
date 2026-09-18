"""FeatureMatrixBuilder — orchestrates the whole pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config.schema import SchemaConfig
from ..io.cohort_cache import CohortDataCache
from ..io.loaders import DuckDBSource
from ..registry.registry import DEFAULT_REGISTRY, PhenotypeRegistry, PhenotypeSpec
from ..temporal.context import EpisodeContext


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

    def build(self, subjects: Optional[List] = None, raise_on_error: bool = False) -> pd.DataFrame:
        episodes = self.source.load_episodes(self.schema.episodes, subjects=subjects)
        if episodes.empty:
            return pd.DataFrame()

        available_domains = set(self.schema.domains.keys())
        domains_needed = set()

        for spec in self.specs:
            for d in spec.domains:
                if d not in available_domains:
                    raise ValueError(
                        f"Configuration Error: Phenotype '{spec.name}' requires domain '{d}', "
                        f"but '{d}' is missing from your schema_config.yaml. "
                        f"Available schema domains: {list(available_domains)}"
                    )
                domains_needed.add(d)

        domains_needed = sorted(domains_needed)
        # ----------------------------

        cache = CohortDataCache(self.schema, source=self.source).preload(
            domains_needed, subjects=episodes["subject"].unique().tolist()
        )

        rows: List[Dict] = []
        for _, ep in episodes.iterrows():
            ctx = EpisodeContext.from_cache(
                cache,
                subject=ep["subject"],
                index_admission=ep["index_admission"],
                domains=domains_needed,
                index_discharge=ep.get("index_discharge"),
            )
            row: Dict = ep.to_dict()

            for spec in self.specs:
                self._run_one(spec, ctx, row, raise_on_error)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Update your index columns to look for the logical names used by the dictionary
        index_cols = [
            col for col in ["spell", "encntr"]
            if col in df.columns
        ]
        return df.set_index(index_cols) if index_cols else df

        #return pd.DataFrame(rows).set_index(["SPELL_IDENTIFIER", "ENCNTR_ID"])

    @staticmethod
    def _run_one(spec: PhenotypeSpec,
                 ctx: EpisodeContext,
                 row: Dict,
                 raise_on_error: bool) -> None:
        try:
            kwargs = getattr(spec, "kwargs", {})
            result = spec.func(ctx, **kwargs)
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
