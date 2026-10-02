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

    def preload(
        self,
        domains: Iterable[str],
        subjects: Optional[List] = None,
        domain_codes: Optional[Dict[str, set]] = None,
        domain_cols: Optional[Dict[str, set]] = None,
    ) -> "CohortDataCache":
        """"""
        domain_codes = domain_codes or {}
        domain_cols = domain_cols or {}

        for domain in sorted(set(domains)):
            table_schema = self.schema.domain(domain)

            codes = domain_codes.get(domain)
            cols = domain_cols.get(domain)

            codes_list = list(codes) if codes else None
            cols_list = list(cols) if cols else None

            df = self.source.load_domain(
                table_schema,
                subjects=subjects,
                codes=codes_list,
                columns=cols_list
            )

            if not df.empty:
                # Important for the context to work
                df = df.sort_values("timestamp")
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
