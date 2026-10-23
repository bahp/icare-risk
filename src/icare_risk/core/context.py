import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from icare_risk.core.schema import ClinicalSchema, TableSchema
from icare_risk.data.dataset import Dataset


class EpisodeContext:
    def __init__(
            self,
            subject_id: int,
            encounter_id: int,
            admission_dt: datetime,
            discharge_dt: datetime,
            dataset: Dataset,
            schema: ClinicalSchema
    ):
        self.subject_id = subject_id
        self.encounter_id = encounter_id
        self.admission_dt = admission_dt
        self.dataset = dataset
        self.schema = schema

    def _execute_pushdown(
            self,
            table_conf,
            columns,
            where_clause: str,
            params: list,
            concept_codes: Optional[list] = None
    ) -> pd.DataFrame:

        cols = ", ".join(columns) if columns else "*"

        # 1. Base filter (Subject)
        base_where = f"{table_conf.subject_col} = ?"
        full_params = [self.subject_id] + params

        # 2. Dynamic Code Filter (with defensive casting)
        code_where = ""
        if concept_codes and table_conf.code_col:
            placeholders = ", ".join(["?"] * len(concept_codes))

            # DEFENSIVE FIX: Cast the column to VARCHAR to ignore DuckDB's guessed types
            code_where = f" AND CAST({table_conf.code_col} AS VARCHAR) IN ({placeholders})"

            # DEFENSIVE FIX: Force all passed codes to strings
            full_params.extend([str(c) for c in concept_codes])

        # 3. Assemble and execute
        query = f"SELECT {cols} FROM {table_conf.table_name} WHERE {base_where} AND {where_clause} {code_where}"

        return self.dataset._execute_query(query, full_params)

    def get_past(
            self, table_attr: str, columns: Optional[List[str]] = None, concept_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetches data strictly before T_admission."""
        table_conf = getattr(self.schema, table_attr)
        where_clause = f"CAST({table_conf.timestamp_col} AS TIMESTAMP) < ?"
        return self._execute_pushdown(table_conf, columns, where_clause, [self.admission_dt], concept_codes)

    def get_window(
            self, table_attr: str, start_h: float, end_h: float,
            columns: Optional[List[str]] = None, concept_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetches data within [T_admission + start_h, T_admission + end_h]."""
        table_conf = getattr(self.schema, table_attr)
        start_t = self.admission_dt + timedelta(hours=start_h)
        end_t = self.admission_dt + timedelta(hours=end_h)
        where_clause = f"CAST({table_conf.timestamp_col} AS TIMESTAMP) >= ? AND CAST({table_conf.timestamp_col} AS TIMESTAMP) <= ?"
        return self._execute_pushdown(table_conf, columns, where_clause, [start_t, end_t], concept_codes)

    def get_current(self, table_attr: str,
                          columns: Optional[List[str]] = None,
                          concept_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetches data tied to the current ENCNTR_ID."""
        table_conf = getattr(self.schema, table_attr)

        # Strategy A: We have an Encounter ID column in this table
        if table_conf.encounter_col:
            where_clauses = [f"{table_conf.encounter_col} = ?"]
            params = [self.encounter_id]

        # Strategy B: No Encounter ID available, fallback to temporal overlap
        else:
            if not self.discharge_dt:
                raise ValueError(
                    f"Cannot temporally bound get_current() for '{table_attr}'. "
                    "Dataset lacks both an encounter_col and a discharge date."
                )
            where_clauses = [
                f"CAST({table_conf.timestamp_col} AS TIMESTAMP) >= ?",
                f"CAST({table_conf.timestamp_col} AS TIMESTAMP) <= ?"
            ]
            params = [self.admission_dt, self.discharge_dt]

        # Apply code pushdown if requested
        if concept_codes and table_conf.code_col:
            placeholders = ", ".join(["?"] * len(concept_codes))
            where_clauses.append(f"{table_conf.code_col} IN ({placeholders})")
            params.extend(concept_codes)

        where_sql = " AND ".join(where_clauses)

        return self._execute_pushdown(table_conf, columns, where_sql, params)