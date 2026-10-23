from dataclasses import dataclass
from typing import Optional

@dataclass
class TableSchema:
    table_name: str
    subject_col: str = "SUBJECT"
    encounter_col: Optional[str] = "ENCNTR_ID"
    timestamp_col: Optional[str] = None
    code_col: Optional[str] = None
    value_col: Optional[str] = None

@dataclass
class ClinicalSchema:
    episodes: TableSchema = TableSchema(
        table_name="episodes",
        encounter_col="ENCNTR_ID",
        timestamp_col="ADMISSION_DATE"
    )
    vitals: TableSchema = TableSchema(
        table_name="vitals",
        timestamp_col="OBSERVATION_PERFORMED_DT",
        code_col="OBSERVATION_CODE",
        value_col="OBSERVATION_RESULT_CLEAN"
    )
    problems: TableSchema = TableSchema(
        table_name="problems",
        timestamp_col="PROBLEM_DT_TM",
        code_col="PROBLEM_CODE"
    )
    prescribing: TableSchema = TableSchema(
        table_name="prescribing",
        timestamp_col="ORDER_DT_TM",
        code_col="MEDICATION_NAME_SHORT",
        value_col="ORDERED_DOSE"
    )
    pathology: TableSchema = TableSchema(
        table_name="pathology",
        timestamp_col="RESULT_AVAILABLE_DT",
        code_col="TEST_NAME",
        value_col="RESULT_CLEANED"
    )