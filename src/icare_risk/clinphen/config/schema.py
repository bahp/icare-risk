"""Schema configuration layer.

The ONE place in the codebase that knows about physical column names. Every
other layer (temporal slicing, phenotype logic, the runner) speaks only in
logical concept names (``value``, ``code``, ``timestamp``, ``drug`` ...).

Swapping to a new hospital extract / research data mart / renamed columns
means editing this file only — zero changes anywhere else.
"""
from __future__ import annotations

import yaml
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class TableSchema:
    """Physical -> logical column mapping for one domain (event-level) table.

    ``source`` may be a parquet path, a csv path, or the name of a
    view/table already registered in the DuckDB connection (e.g. a
    lakehouse catalog entry, or an in-memory frame registered for testing).
    """

    source: str
    subject: str
    timestamp: str
    mapping: Dict[str, str] = field(default_factory=dict)  # logical -> physical

    @property
    def code_col(self) -> Optional[str]:
        """Returns the physical column name mapped to logical 'code'."""
        return self.mapping.get("code")

    def select_columns(self) -> Dict[str, str]:
        """logical_name -> physical_name, always including subject & timestamp."""
        cols = {"subject": self.subject, "timestamp": self.timestamp}
        cols.update(self.mapping)
        return cols


@dataclass(frozen=True)
class EpisodeTableSchema:
    """"""
    source: str
    subject: str
    encounter: str
    admission_date: str
    admission_time: Optional[str] = None
    discharge_date: Optional[str] = None
    spell: Optional[str] = None
    mapping: Dict[str, str] = field(default_factory=dict)

    def normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the dataframe.

        1. Translates physical column names to logical names.
        2. uilds index_admission/index_discharge.
        """
        df = df.copy()

        # 1. Build physical -> logical mapping
        rename_map = {
            self.subject: "subject",
            self.encounter: "encntr",
            self.admission_date: "admission_date",
        }
        if self.spell and self.spell in df.columns:
            rename_map[self.spell] = "spell"
        if self.admission_time and self.admission_time in df.columns:
            rename_map[self.admission_time] = "admission_time"
        if self.discharge_date and self.discharge_date in df.columns:
            rename_map[self.discharge_date] = "discharge_date"

        # Map custom domain fields
        for logical_col, physical_col in self.mapping.items():
            if physical_col in df.columns:
                rename_map[physical_col] = logical_col

        df = df.rename(columns=rename_map)

        # 2. Build index_admission if missing
        if "index_admission" not in df.columns:
            if "admission_time" in df.columns and "admission_date" in df.columns:
                combined = df["admission_date"].astype(str) + " " + df["admission_time"].astype(str)
                df["index_admission"] = pd.to_datetime(combined, errors="coerce")
            elif "admission_date" in df.columns:
                df["index_admission"] = pd.to_datetime(df["admission_date"], errors="coerce")

        # 3. Build index_discharge if missing
        if "index_discharge" not in df.columns:
            if "discharge_date" in df.columns:
                df["index_discharge"] = pd.to_datetime(df["discharge_date"], errors="coerce")
            else:
                df["index_discharge"] = pd.NaT

        return df

@dataclass(frozen=True)
class SchemaConfig:
    episodes: EpisodeTableSchema
    domains: Dict[str, TableSchema]

    def domain(self, name: str) -> TableSchema:
        try:
            return self.domains[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown domain table '{name}'. Registered domains: {list(self.domains)}"
            ) from exc


def load_schema_from_yaml(yaml_path: str) -> SchemaConfig:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Unpack episodes config
    episodes_data = config.get('episodes', {})
    episodes = EpisodeTableSchema(**episodes_data)

    # Unpack domains config
    domains = {}
    for domain_name, domain_data in config.get('domains', {}).items():
        domains[domain_name] = TableSchema(**domain_data)

    return SchemaConfig(episodes=episodes, domains=domains)

# ---------------------------------------------------------------------------
# Example environment configuration matching the schema described in the
# architecture brief. Point `source` at real parquet/csv paths (or catalog
# view names) per-deployment; nothing else in the package needs to change.
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA = SchemaConfig(
    episodes=EpisodeTableSchema(
        source="data/episodes.parquet",
        subject="SUBJECT",
        spell="SPELL_IDENTIFIER",
        encounter="ENCNTR_ID",
        admission_date="ADMISSION_DATE",
        admission_time="ADMISSION_TIME",
        discharge_date="DISCHARGE_DATE",
        mapping={"age_at_admission": "AGE_AT_ADMISSION", "specialty": "MAIN_SPECIALTY_CODE"},
    ),
    domains={
        "microbiology": TableSchema(
            source="data/microbiology.parquet",
            subject="SUBJECT",
            timestamp="COLLECTION_DT_TM",
            mapping={
                "code": "ORDER_CODE",
                "site": "SITE",
                "organism": "ORGANISM_BUG",
                "sensitivity": "SENSITIVITY",
            },
        ),
        "pathology": TableSchema(
            source="data/pathology.parquet",
            subject="SUBJECT",
            timestamp="SAMPLE_COLLECTED_DT",
            mapping={
                "code": "TEST_CODE",
                "name": "TEST_NAME",
                "value": "RESULT_CLEANED",
                "lower_range": "RESULT_LOWER_RANGE",
                "upper_range": "RESULT_UPPER_RANGE",
            },
        ),
        "prescribing": TableSchema(
            source="data/prescribing.parquet",
            subject="SUBJECT",
            timestamp="ORDER_DT_TM",
            mapping={
                "drug": "MEDICATION_NAME",
                "drug_short": "MEDICATION_NAME_SHORT",
                "therapeutic_class": "THERAPEUTICAL_CLASS",
                "dose": "ORDERED_DOSE",
            },
        ),
        "problems": TableSchema(
            source="data/problems.parquet",
            subject="SUBJECT",
            timestamp="PROBLEM_DT_TM",
            mapping={"code": "PROBLEM_CODE", "description": "PROBLEM_DESC"},
        ),
        "vitals": TableSchema(
            source="data/vitals.parquet",
            subject="SUBJECT",
            timestamp="OBSERVATION_PERFORMED_DT",
            mapping={
                "code": "OBSERVATION_CODE",
                "name": "OBSERVATION_NAME",
                "value": "OBSERVATION_RESULT_CLEAN",
            },
        ),
    },
)
