"""EpisodeContext — the single, leakage-safe data access surface for phenotypes."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..io.cohort_cache import CohortDataCache
from ..utils.casting import normalize_frame


def _parse_offset(token: str) -> pd.Timedelta:
    return pd.Timedelta(token)


class EpisodeContext:
    def __init__(
        self,
        subject,
        index_admission: pd.Timestamp,
        tables: Dict[str, pd.DataFrame],
        index_discharge: Optional[pd.Timestamp] = None,
    ):
        self.subject = subject
        self.index_admission = pd.Timestamp(index_admission)
        self.index_discharge = pd.Timestamp(index_discharge) if pd.notna(index_discharge) else None
        self._tables = tables
        # Add caches to avoid redundant slicing and indexing
        self._historical_cache: Dict[Tuple, pd.DataFrame] = {}
        self._current_cache: Dict[Tuple, pd.DataFrame] = {}

    @classmethod
    def from_cache(
        cls,
        cache: CohortDataCache,
        subject,
        index_admission,
        domains: List[str],
        index_discharge=None,
    ) -> "EpisodeContext":
        tables = {domain: cache.get(domain, subject) for domain in domains}
        return cls(subject, index_admission, tables, index_discharge=index_discharge)

    @classmethod
    def from_frames(
        cls,
        subject,
        index_admission,
        frames: Dict[str, pd.DataFrame],
        index_discharge=None,
    ) -> "EpisodeContext":
        tables = {name: normalize_frame(df) for name, df in frames.items()}
        return cls(subject, index_admission, tables, index_discharge=index_discharge)

    def get_historical(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        strictly_before: bool = True,
    ) -> pd.DataFrame:
        """
        df = self._tables.get(table)
        if df is None or df.empty:
            return pd.DataFrame(columns=columns or [])
        mask = (
            df["timestamp"] < self.index_admission
            if strictly_before
            else df["timestamp"] <= self.index_admission
        )
        return self._project(df.loc[mask], columns)
        """
        # Create a stable cache key
        cache_key = (table, tuple(columns) if columns else None, strictly_before)
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        df = self._tables.get(table)
        if df is None or df.empty:
            res = pd.DataFrame(columns=columns or [])
            self._historical_cache[cache_key] = res
            return res

        mask = (
            df["timestamp"] < self.index_admission
            if strictly_before
            else df["timestamp"] <= self.index_admission
        )
        res = self._project(df.loc[mask], columns)
        self._historical_cache[cache_key] = res
        return res

    def get_current(self,
                    table: str,
                    window: Optional[Tuple[str, str]] = None,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        df = self._tables.get(table)
        if df is None or df.empty:
            return pd.DataFrame(columns=columns or [])
        if window is None:
            return self._project(df, columns)
        start = self.index_admission + _parse_offset(window[0]) if window[0] else None
        end = self.index_admission + _parse_offset(window[1]) if window[1] else None
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= (df["timestamp"] >= start)
        if end is not None:
            mask &= (df["timestamp"] < end)
        return self._project(df.loc[mask], columns)
        """
        cache_key = (table, window, tuple(columns) if columns else None)
        if cache_key in self._current_cache:
            return self._current_cache[cache_key]

        df = self._tables.get(table)
        if df is None or df.empty:
            res = pd.DataFrame(columns=columns or [])
            self._current_cache[cache_key] = res
            return res

        if window is None:
            res = self._project(df, columns)
            self._current_cache[cache_key] = res
            return res

        start = self.index_admission + _parse_offset(window[0]) if window[0] else None
        end = self.index_admission + _parse_offset(window[1]) if window[1] else None
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= (df["timestamp"] >= start)
        if end is not None:
            mask &= (df["timestamp"] < end)

        res = self._project(df.loc[mask], columns)
        self._current_cache[cache_key] = res
        return res

    get_relative = get_current

    @staticmethod
    def _project(df: pd.DataFrame, columns: Optional[List[str]]) -> pd.DataFrame:
        if not columns:
            return df.reset_index(drop=True)
        keep = [c for c in ["subject", "timestamp", *columns] if c in df.columns]
        return df.loc[:, keep].reset_index(drop=True)
