"""DuckDB access layer: column projection + subject-list pushdown."""
from __future__ import annotations

from typing import Iterable, List, Optional

import duckdb
import pandas as pd

from ..config.schema import EpisodeTableSchema, TableSchema
from ..utils.casting import normalize_frame


def _resolve_source_sql(source: str) -> str:
    lowered = source.lower()
    if lowered.endswith(".parquet"):
        return f"read_parquet('{source}')"
    if lowered.endswith(".csv"):
        return f"read_csv_auto('{source}')"
    return source


class DuckDBSource:
    """Thin, swappable DuckDB access layer."""

    def __init__(self, connection: Optional["duckdb.DuckDBPyConnection"] = None):
        self.con = connection or duckdb.connect(database=":memory:")

    def load_domain(
        self,
        table_schema: TableSchema,
        subjects: Optional[Iterable] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        all_cols = table_schema.select_columns()
        wanted = {"subject", "timestamp", *(columns or all_cols.keys())}
        select_parts = [
            f'"{phys}" AS "{logical}"' for logical, phys in all_cols.items() if logical in wanted
        ]
        src_sql = _resolve_source_sql(table_schema.source)
        sql = f'SELECT {", ".join(select_parts)} FROM {src_sql}'

        params: list = []
        if subjects is not None:
            sql += f' WHERE "{table_schema.subject}" IN (SELECT * FROM UNNEST(?))'
            params.append(list(subjects))

        df = self.con.execute(sql, params).fetchdf()
        return normalize_frame(df)

    def load_episodes(
        self,
        episode_schema: EpisodeTableSchema,
        subjects: Optional[Iterable] = None,
    ) -> pd.DataFrame:
        select_parts = [
            f'"{episode_schema.subject}" AS "subject"',
            f'"{episode_schema.encounter}" AS "encntr"',
            f'"{episode_schema.admission_date}" AS "admission_date"',
        ]
        if episode_schema.spell not in (None, "None", ""):
            select_parts.append(f'"{episode_schema.spell}" AS "spell"')
        if episode_schema.admission_time:
            select_parts.append(f'"{episode_schema.admission_time}" AS "admission_time"')
        if episode_schema.discharge_date:
            select_parts.append(f'"{episode_schema.discharge_date}" AS "discharge_date"')
        for logical, phys in episode_schema.mapping.items():
            select_parts.append(f'"{phys}" AS "{logical}"')

        src_sql = _resolve_source_sql(episode_schema.source)
        sql = f'SELECT {", ".join(select_parts)} FROM {src_sql}'

        params: list = []
        if subjects is not None:
            sql += f' WHERE "{episode_schema.subject}" IN (SELECT * FROM UNNEST(?))'
            params.append(list(subjects))

        df = self.con.execute(sql, params).fetchdf()

        if "admission_time" in df.columns:
            combined = df["admission_date"].astype(str) + " " + df["admission_time"].astype(str)
            df["index_admission"] = pd.to_datetime(combined, errors="coerce")
        else:
            df["index_admission"] = pd.to_datetime(df["admission_date"], errors="coerce")

        if "discharge_date" in df.columns:
            df["index_discharge"] = pd.to_datetime(df["discharge_date"], errors="coerce")
        else:
            df["index_discharge"] = pd.NaT

        return df
