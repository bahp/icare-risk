"""DuckDB access layer: column projection + subject-list pushdown."""
from __future__ import annotations

from typing import Iterable, List, Optional, Union
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
    """Thin, performant DuckDB access layer with SQL pushdown."""

    def __init__(self, connection: Optional["duckdb.DuckDBPyConnection"] = None):
        self.con = connection or duckdb.connect(database=":memory:")

    def load_domain(
        self,
        table_schema: TableSchema,
        subjects: Optional[Iterable] = None,
        codes: Optional[Iterable[Union[str, int]]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
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

        """
        src_sql = _resolve_source_sql(table_schema.source)
        where_clauses = []
        params = []

        load_all = columns is not None and "*" in columns

        if load_all:
            cols_sql = "*"
        else:
            # 1. Map Logical Names directly to Physical Names for SQL Aliasing
            select_map = {
                "subject": table_schema.subject,
                "timestamp": table_schema.timestamp,
            }

            for logical, physical in table_schema.mapping.items():
                select_map[logical] = physical

            # Include any explicitly requested extra physical columns
            if columns:
                for col in columns:
                    if col not in select_map:
                        select_map[col] = col

            # Build SELECT clause with aliasing: "OBSERVATION_CODE" AS "code"
            cols_sql = ", ".join(f'"{phys}" AS "{log}"' for log, phys in select_map.items())

        sql = f"SELECT {cols_sql} FROM {src_sql}"

        # Note: Both sides cast to string to prevent BIGINT/VARCHAR issue.
        # 2. Pushdown Subject Filter (Must reference physical name)
        if subjects is not None:
            where_clauses.append(f'CAST("{table_schema.subject}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
            params.append([str(s) for s in subjects])

        # 3. Pushdown Code Filter (Must reference physical name)
        if codes is not None and table_schema.code_col:
            where_clauses.append(f'CAST("{table_schema.code_col}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
            params.append([str(c) for c in codes])

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # 4. Fetch Execution
        df = self.con.execute(sql, params).fetchdf()
        if df.empty:
            return df

        # 5. Timestamp Casting and Fallback Aliasing (if columns=["*"] was used)
        if load_all:
            df["subject"] = df[table_schema.subject]
            df["timestamp"] = pd.to_datetime(df[table_schema.timestamp], errors="coerce")
            for logical, physical in table_schema.mapping.items():
                if physical in df.columns and logical not in df.columns:
                    df[logical] = df[physical]
        else:
            # If not load_all, the DataFrame is already perfectly aliased by DuckDB
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df


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
