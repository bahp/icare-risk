"""EpisodeContext — the single, leakage-safe data access surface for phenotypes."""
from __future__ import annotations

import pandas as pd

from typing import Dict, List, Optional, Tuple
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
        #self._historical_cache: Dict[Tuple, pd.DataFrame] = {}
        #self._current_cache: Dict[Tuple, pd.DataFrame] = {}
        self._window_cache: Dict[Tuple, pd.DataFrame] = {}

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

    def get_historicalv3(
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

    def get_currentv3(self,
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

    def get_window(
        self,
        table: str,
        window: Optional[Tuple[Optional[str], Optional[str]]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieves records for a specific domain within a relative time window.

        Parameters
        ----------
        table : str
            The target domain table name.
        window : tuple of str, optional
            Relative time offsets (start, end) around index_admission (e.g. ("-90d", "0h")).
            If None, returns all available records for the table without time filtering.
        columns : list of str, optional
            Subset of columns to project from the target table.
        """
        # 1. Cast window to a tuple to ensure the cache key is hashable
        safe_window = tuple(window) if isinstance(window, list) else window
        cache_key = (table, safe_window, tuple(columns) if columns else None)

        # 2. Use safe_window in the cache key
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        df = self._tables.get(table)
        if df is None or df.empty:
            res = pd.DataFrame(columns=columns or [])
            self._window_cache[cache_key] = res
            return res

        if window is None:
            res = self._project(df, columns)
            self._window_cache[cache_key] = res
            return res

        start_offset, end_offset = safe_window
        start = self.index_admission + _parse_offset(start_offset) if start_offset else None
        end = self.index_admission + _parse_offset(end_offset) if end_offset else None

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= (df["timestamp"] >= start)
        if end is not None:
            mask &= (df["timestamp"] < end)

        res = self._project(df.loc[mask], columns)
        self._window_cache[cache_key] = res
        return res

    def get_all(self, table: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieves all records for the patient regardless of timestamps."""
        return self.get_window(table, window=None, columns=columns)

    def get_historical(
            self,
            table: str,
            columns: Optional[List[str]] = None,
            strictly_before: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieves records occurring before the index admission.
        """
        # (None, "0h") means unbounded past up to the exact moment of index_admission
        # If strictly_before is False, we add a 1-second buffer to make it inclusive (<= admission)
        end_boundary = "0h" if strictly_before else "1s"
        return self.get_window(table, window=(None, end_boundary), columns=columns)

    def get_current(
            self,
            table: str,
            columns: Optional[List[str]] = None,
            window: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Retrieves records for the current admission.
        If no window is specified, retrieves data from admission up to discharge.
        If no discharge timestamp exists, retrieves all data from admission onwards (up to current date).
        """
        if window is not None:
            return self.get_window(table, window=window, columns=columns)

        # Unbounded future starting exactly at index_admission
        df_onwards = self.get_window(table, window=("0h", None), columns=columns)

        # If a discharge date is recorded, clip the data at discharge
        if self.index_discharge and not df_onwards.empty:
            mask = df_onwards["timestamp"] <= self.index_discharge
            return df_onwards.loc[mask].reset_index(drop=True)

        return df_onwards


    @staticmethod
    def _project(df: pd.DataFrame, columns: Optional[List[str]]) -> pd.DataFrame:
        if not columns:
            return df.reset_index(drop=True)
        keep = [c for c in ["subject", "timestamp", *columns] if c in df.columns]
        return df.loc[:, keep].reset_index(drop=True)
