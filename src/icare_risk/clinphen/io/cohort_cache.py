"""Bulk cohort loading with O(1) per-subject lookup."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config.schema import SchemaConfig
from .loaders import DuckDBSource


class CohortDataCache:
    def __init__(self, schema: SchemaConfig, source: Optional[DuckDBSource] = None):
        self.schema = schema
        self.source = source or DuckDBSource()
        self._by_domain_subject: Dict[str, Dict[object, pd.DataFrame]] = {}

    def preload(self, domains: Iterable[str], subjects: Optional[List] = None) -> "CohortDataCache":
        for domain in sorted(set(domains)):
            table_schema = self.schema.domain(domain)
            df = self.source.load_domain(table_schema, subjects=subjects)
            df = df.sort_values("timestamp") # important for the context to work.

            if "code" in df.columns:
                df["code"] = df["code"].astype("string").str.upper()

            self._by_domain_subject[domain] = {
                subj: sub_df.reset_index(drop=True)
                for subj, sub_df in df.groupby("subject", sort=False)
            }
        return self

    def get(self, domain: str, subject) -> pd.DataFrame:
        subj_map = self._by_domain_subject.get(domain, {})
        df = subj_map.get(subject)
        if df is None:
            return pd.DataFrame(columns=["subject", "timestamp"])
        return df
