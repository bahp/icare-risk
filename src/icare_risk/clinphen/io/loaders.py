"""DuckDB access layer: column projection + subject-list pushdown."""
from __future__ import annotations

from typing import Iterable, List, Optional, Union, Dict, Any
import duckdb
import pandas as pd

from ..config.schema import EpisodeTableSchema, TableSchema
from ..utils.casting import normalize_frame


def get_total_memory_usage(df):
    total_bytes = df.memory_usage(deep=True).sum()
    total_mb = total_bytes / (1024 ** 2)
    return total_mb

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
        filters: Optional[Dict[str, Iterable[Any]]] = None,
        skip_mapping: bool = False,
        case_sensitive: bool = False
    ) -> pd.DataFrame:

        src_sql = _resolve_source_sql(table_schema.source)

        mapping = table_schema.mapping
        excluded = set(table_schema.exclude_columns)
        
        # 1. Build SELECT clause (Mapped vs Raw)
        if skip_mapping or (columns and "*" in columns):
            cols_sql = "*"
        else:
            target_cols = columns if columns else list(mapping.keys())
            
            select_parts = [
                f'"{table_schema.subject}" AS subject',
                f'"{table_schema.timestamp}" AS timestamp'
            ]
            
            for col in target_cols:
                if col in excluded or col in ("subject", "timestamp"):
                    continue
                phys_col = mapping.get(col, col)
                if phys_col == col:
                    select_parts.append(f'"{phys_col}"')
                else:
                    select_parts.append(f'"{phys_col}" AS "{col}"')
                    
            cols_sql = ", ".join(select_parts)

        sql = f"SELECT {cols_sql} FROM {src_sql}"

        # 2. Build WHERE clause (Combine defaults, filters, subjects, and codes)
        where_clauses = []
        params = []
        
        active_filters = dict(table_schema.filters) if not skip_mapping else {}
        if filters:
            active_filters.update(filters)

        if subjects is not None:
            active_filters["subject"] = subjects
            
        if codes is not None:
            code_col = table_schema.code_col or mapping.get("code", "code")
            active_filters[code_col] = codes

        for key, val_list in active_filters.items():
            if not val_list:
                continue
            
            # Resolve logical names to physical column names safely
            if key in ("subject", "SUBJECT"):
                phys_col = table_schema.subject
            elif key in ("timestamp", "TIMESTAMP"):
                phys_col = table_schema.timestamp
            else:
                phys_col = mapping.get(key, key)

            if not case_sensitive:
                where_clauses.append(f'CAST(UPPER("{phys_col}") AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
                params.append([str(v).upper() for v in val_list])
            else:
                where_clauses.append(f'CAST("{phys_col}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
                params.append([str(v) for v in val_list])

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # 3. Execute and Post-process
        df = self.con.execute(sql, params).fetchdf()
        if df.empty:
            return df

        if not skip_mapping:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "subject" in df.columns:
                df["subject"] = df["subject"].astype(str)
        else:
            ts_phys = table_schema.timestamp
            if ts_phys in df.columns:
                df[ts_phys] = pd.to_datetime(df[ts_phys], errors="coerce")

        return df

    def load_domain2(
        self,
        table_schema: TableSchema,
        subjects: Optional[Iterable] = None,
        codes: Optional[Iterable[Union[str, int]]] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Iterable[Any]]] = None,
        skip_mapping: bool = False
    ) -> pd.DataFrame:
        """

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

            # 2. Add those in mapping (excluding those in exclude columns)
            excluded = set(table_schema.exclude_columns)
            for logical, physical in table_schema.mapping.items():
                if logical not in excluded and physical not in excluded:
                    select_map[logical] = physical

            # 3. If user request an specific column, respect them
            if columns:
                for col in columns:
                    if col not in select_map:
                        select_map[col] = col


            # Build SELECT clause with aliasing: "OBSERVATION_CODE" AS "code"
            cols_sql = ", ".join(f'"{phys}" AS "{log}"' for log, phys in select_map.items())

        sql = f"SELECT {cols_sql} FROM {src_sql}"

        # Note: Both sides cast to string to prevent BIGINT/VARCHAR issue.
        # 4. Pushdown Subject Filter (Must reference physical name)
        if subjects is not None:
            where_clauses.append(f'CAST("{table_schema.subject}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
            params.append([str(s) for s in subjects])

        # 5. Pushdown Code Filter (Must reference physical name)
        if codes is not None and table_schema.code_col:
            phys_code_col = table_schema.code_col  # Gets physical name (e.g. 'PROBLEM_CODE')
            where_clauses.append(f'CAST("{phys_code_col}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
            params.append([str(c) for c in codes])

        # 6. Combine schema-level default filters with runtime overrides
        active_filters = {**table_schema.filters}
        if filters:
            active_filters.update(filters)  # Runtime filters take precedence

        if active_filters:
            for col_name, val_list in active_filters.items():
                if val_list:
                    phys_col = table_schema.mapping.get(col_name, col_name)
                    if phys_col == "subject":
                        phys_col = table_schema.subject
                    elif phys_col == "timestamp":
                        phys_col = table_schema.timestamp

                    where_clauses.append(f'CAST("{phys_col}" AS VARCHAR) IN (SELECT * FROM UNNEST(?))')
                    params.append([str(v) for v in val_list])

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

            # Respect schema-level column exclusions even during wildcard loads
            if table_schema.exclude_columns:
                cols_to_drop = [
                    col for col in table_schema.exclude_columns 
                    if col in df.columns
                ]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
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



from typing import Dict, List, Optional, Any
import pandas as pd

def load_pipeline_datasets(
    source: DuckDBSource,
    schema: SchemaConfig,
    registry: Any,
    domains_to_load: Optional[List[str]] = None,
    skip_mapping: bool = True,
    case_sensitive: bool = False,
    custom_codes: Optional[Dict[str, list]] = None,
    verbose: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Automatically loads selected domains using global dependencies from 
    the phenotype registry.

    Parameters
    ----------
    source : DuckDBSource
        DuckDBSource instance.
    schema : SchemaConfig
        SchemaConfig instance.
    registry : Any
        Phenotype registry instance.
    domains_to_load : list of str, optional
        List of domain names to load (e.g., ['episodes', 'problems', ...]).
        Defaults to core domains if None.
    skip_mapping : bool, default=True
        Whether to skip column aliasing (default True for raw comparison).
    custom_codes : dict of list, optional
        Optional dictionary overriding registry codes per domain.
    case_sensitive : bool, default=False
        Whether filter criteria and code matching should respect upper/lowercase.
    verbose : int, default=1
        Verbosity printing level.

    Returns
    -------
    dict of str to pd.DataFrame
        A dictionary mapping each requested domain name to its loaded DataFrame.
    """
    if domains_to_load is None:
        raise ValueError("Parameter 'domains_to_load' must be specified and cannot be None.")

    # 1. Validate that all requested domains exist in schema or are 'episodes'
    available_domains = set(schema.domains.keys())
    available_domains.add("episodes")
    
    invalid_domains = [d for d in domains_to_load if d not in available_domains]
    if invalid_domains:
        raise ValueError(
            f"The following requested domain(s) do not exist in the schema: {invalid_domains}. "
            f"Available domains are: {sorted(list(available_domains))}"
        )


    required_codes, required_cols = registry.get_global_dependencies()
    custom_codes = custom_codes or {}
        
    datasets = {}
    
    # 1. Load Episodes separately (uses EpisodeTableSchema & normalization)
    if "episodes" in domains_to_load:
        query = f"SELECT * FROM read_parquet('{schema.episodes.source}')"
        df_ep = source.con.execute(query).fetchdf()
        #df_ep = schema.episodes.normalize_frame(episodes_raw)
        
        if "ADMISSION_DATE" in df_ep.columns:
            df_ep["ADMISSION_DATE"] = pd.to_datetime(df_ep["ADMISSION_DATE"], errors="coerce")
        if "DISCHARGE_DATE" in df_ep.columns:
            df_ep["DISCHARGE_DATE"] = pd.to_datetime(df_ep["DISCHARGE_DATE"], errors="coerce")
        datasets["episodes"] = df_ep

        if verbose > 0:
            print(f"Loaded... episodes ({get_total_memory_usage(df_ep):.2f}MB) shape {df_ep.shape}")
        
    # 2. Load Event Domains dynamically using load_domain() and registry pushdowns
    for domain_name in domains_to_load:
        if domain_name == "episodes":
            continue
            
        if domain_name in schema.domains:
            # Fallback priority: Custom override -> Registry extracted codes -> None
            codes = custom_codes.get(domain_name) or required_codes.get(domain_name)
            cols = required_cols.get(domain_name)
            
            df = source.load_domain(
                schema.domains[domain_name],
                skip_mapping=skip_mapping,
                codes=codes,
                columns=list(cols) if cols else None,
                case_sensitive=case_sensitive
            )
            
            # Normalize timestamp datetime parsing automatically
            for ts_col in ["PROBLEM_DT_TM", "DIAGNOSIS_DATE", "ORDER_DT_TM", "timestamp"]:
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                    
            datasets[domain_name] = df

            if verbose > 0:
                print(f"Loaded... {domain_name} ({get_total_memory_usage(df):.2f}MB) shape {df.shape} | Filtered codes: {len(codes) if codes else 'None'}")

    return datasets


def load_from_duckdb(
    schema_path: Optional[str] = None,
    phenotype_path: Optional[str] = None,
    domains_to_load: Optional[List[str]] = None,
    custom_codes: Optional[Dict[str, list]] = None,
    skip_mapping: bool = True,
    case_sensitive: bool = False,
    verbose: int = 1
) -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Helper function to load configured datasets from DuckDB using schema 
    and phenotype specifications.

    Parameters
    ----------
    schema_path : str, optional
        Optional path or identifier for the schema YAML config. 
        Defaults to 'configs/schema.yaml'.
    phenotype_path : str, optional
        Optional path or identifier for the phenotypes YAML config. 
        Defaults to 'configs/phenotypes.yaml'.
    domains_to_load : list of str, optional
        List of domain names to load (e.g., ["episodes", "problems", ...]).
        Defaults to core domains if None.
    custom_codes : dict of list, optional
        Optional dictionary overriding registry codes per domain.
    skip_mapping : bool, default=True
        Whether to skip column aliasing (default True).
    case_sensitive : bool, default=False
        Whether filter criteria and code matching should respect upper/lowercase.
    verbose : int, default=1
        Verbosity level for print statements.

    Returns
    -------
    tuple of pd.DataFrame or None
        A tuple of DataFrames corresponding to the requested domains in `domains_to_load`.
    """
    import duckdb
    from icare_risk.clinphen import DuckDBSource
    from icare_risk.clinphen.io.loaders import load_pipeline_datasets
    from icare_risk.clinphen.config.schema import load_schema_from_yaml
    from icare_risk.clinphen.registry.registry import load_phenotypes_from_yaml
    from icare_risk.utils.paths import get_path

    # A. Define configuration paths with sensible defaults
    schema_cfg = get_path(schema_path) 
    phenotype_cfg = get_path(phenotype_path)

    # B. Initialize infrastructure
    schema = load_schema_from_yaml(schema_cfg)
    registry = load_phenotypes_from_yaml(phenotype_cfg)
    source = DuckDBSource(connection=duckdb.connect())

    # C. Load datasets using pipeline loader
    dataframes = load_pipeline_datasets(
        source=source,
        schema=schema,
        registry=registry,
        domains_to_load=domains_to_load,
        skip_mapping=skip_mapping,
        custom_codes=custom_codes,
        case_sensitive=case_sensitive,
        verbose=verbose
    )

    # D. Dynamically return dataframes
    return dataframes