import csv
import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from datetime import datetime
from typing import Iterator
from sqlalchemy import MetaData, select, Table, any_
from sqlalchemy.dialects.postgresql import array

# -------------------------
# Constants
# -------------------------
DEFAULT_SNOWFLAKE_CONFIG_PATH = '/opt/ich/python-snowflake-defaults.json'
DEFAULT_SNOWFLAKE_DATABASE = 'ICHT_PROD'
DEFAULT_SNOWFLAKE_SCHEMA = 'ICARE_ICHT'

# -------------------------
# Connection
# -------------------------
def get_snowflake_engine(database: str = DEFAULT_SNOWFLAKE_DATABASE,
                         schema: str = DEFAULT_SNOWFLAKE_SCHEMA,
                         config_path: str = DEFAULT_SNOWFLAKE_CONFIG_PATH):
    """Generic function to initialize a Snowflake SQLAlchemy engine."""
    from snowflake.connector.pandas_tools import pd_writer
    from snowflake.sqlalchemy import URL
    from sqlalchemy import create_engine
    import json

    with open(config_path) as f:
        conn_string = json.load(f)

    conn_string["database"] = database
    conn_string["schema"] = schema
    engine = create_engine(URL(**conn_string))

    # Return
    return engine


def test_snowflake_connection(engine, test_query: str = "SELECT current_version()"):
    """Generic function to test a Snowflake connection with a provided query."""
    from sqlalchemy import text

    with engine.connect() as connection:
        sql_query = text(test_query)
        result = connection.execute(sql_query).fetchone()
        print(f"Result: {result}")

    print("Login completed!")

def test_snowflake_connection_v2(engine):
    """
    """
    from sqlalchemy import text
    with engine.connect() as connection:
        sql_query = text("SELECT COUNT(*) FROM ICHT_PROD.ICARE_ICHT.ICARE_PROBLEMS_ANON e")
        result = connection.execute(sql_query)
        print(result)
    print("Login completed!")




# -------------------------
# Fetchers
# -------------------------
def fetch_filtered_records_stream(
        engine,
        table_name: str,
        id_list: list,
        target_codes: list = None,
        columns: list = None,
        id_col: str = 'subject',
        code_col: str = 'code'
):
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    # 1. Column Pushdown: Select only required columns
    selected_cols = [table.c[col] for col in columns] if columns else [table]
    stmt = select(*selected_cols).where(table.c[id_col].in_(id_list))

    # 2. Code Pushdown: Filter codes directly in Snowflake query
    if target_codes and code_col in table.columns:
        stmt = stmt.where(table.c[code_col].in_(target_codes))

    compiled_sql = str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute(compiled_sql)

        for batch_num, batch_df in enumerate(cursor.fetch_pandas_batches(), start=1):
            if verbose:
                print(f"Downloaded batch {batch_num} ({len(batch_df):,} rows.)")
            yield batch_df

    finally:
        cursor.close()
        raw_conn.close()

def fetch_records_stream(engine,
                         table_name: str,
                         id_list: list,
                         column_name: str = 'subject',
                         verbose: bool = True) -> Iterator[pd.DataFrame]:
    """Executes query and yields Arrow-downloaded batches as panda dataframes.

    It is designed for memory-safe streaming of large datasets by utilizing
    micro-partition pruning (via a single IN clause) and yielding data
    in manageable chunks rather than loading it all into RAM.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy engine connected to the Snowflake database.
    table_name : str
        The name of the database table to query.
    id_list : list
        A list of identifiers used to filter the table (applied via an IN clause).
    column_name : str, optional
        The column name to apply the ID filter against. Default is 'subject'.
    verbose : bool, optional
        If True, prints progress and batch size information to standard output.
        Default is True.

    Yields
    ------
    pandas.DataFrame
        A batch of queried records as a pandas DataFrame.

    Raises
    ------
    ValueError
        If no valid IDs are provided in `id_list`, or if the specified `column_name`
        does not exist within the target table's schema.
    """
    ids = list(set(x for x in id_list if pd.notna(x)))
    if not ids:
        raise ValueError("Warning: No valid IDs provided for {table_name}.")

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    if column_name not in table.columns:
        raise ValueError(f"Column '{column_name}' does not exist in '{table_name}'.")
    if verbose:
        print(f" -> Compiling query for '{table_name}' across {len(ids):,} IDs...")

    stmt = select(table).where(table.c[column_name].in_(ids))
    compiled_sql = str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute(compiled_sql)

        for batch_num, batch_df in enumerate(cursor.fetch_pandas_batches(), start=1):
            if verbose:
                print(f"Downloaded batch {batch_num} ({len(batch_df):,} rows.)")
            yield batch_df

    finally:
        cursor.close()
        raw_conn.close()


def download_and_save_tables(tables: list,
                             output_dir: str | Path = './data/raw',
                             file_format: str = 'parquet',
                             verbose: bool = True,
                             **fetch_kwargs):
    """
    Downloads tables in batches and saves them locally.

    Iterates through a list of table configurations, fetches the data in
    batches using `fetch_records_stream`, and appends them to files in a
    timestamped output directory. Supports either 'parquet' or 'csv'.

    Parameters
    ----------
    tables : list of dict
        A list of dictionaries configuring the tables to download. Each dictionary
        must contain at least a 'table_name' key.
    output_dir : str or pathlib.Path, optional
        The base output directory where a new timestamped folder will be created.
        Default is './data/raw'.
    file_format : str, optional
        The format to save the downloaded data ('parquet' or 'csv'). Default is 'parquet'.
    verbose : bool, optional
        If True, prints download progress and status messages to standard output.
        Default is True.
    **fetch_kwargs
        Additional keyword arguments to pass directly to `fetch_records_stream`
        (e.g., `id_list`, `column_name`).

    Returns
    -------
    pathlib.Path
        The path to the newly created timestamped directory containing the saved files.

    Raises
    ------
    ValueError
        If the specified `file_format` is not supported (i.e., not 'parquet' or 'csv').
    """

    if file_format not in ['parquet', 'csv']:
        raise ValueError("file_format must be either 'parquet' or 'csv'.")

    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    data_dir = Path(output_dir) / date_str
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting download for {len(tables)} tables...")
    print(f"Saving files to: {data_dir}")

    for item in tables:
        table_name = item.pop('table_name')

        print(f"Fetching '{table_name}'...")

        file_path = data_dir / f"{table_name}.{file_format}"
        writer = None
        total_rows = 0

        try:
            data_stream = fetch_records_stream(
                table_name=table_name, verbose=verbose, **fetch_kwargs
            )
            for i, batch_df in enumerate(data_stream):
                if file_format == 'csv':
                    write_header = (i == 0)
                    batch_df.to_csv(file_path, mode='a', index=False, header=write_header)
                elif file_format == 'parquet':
                    arrow_table = pa.Table.from_pandas(batch_df)
                    if writer is None:
                        writer = pq.ParquetWriter(file_path, arrow_table.schema)
                    writer.write_table(arrow_table)
                total_rows += len(batch_df)

            if total_rows == 0:
                print(f" -> Warning: No records found for '{table_name}'")
            elif verbose:
                print(f" -> Successfully saved {total_rows:,} total rows to {file_path.name}\n")
        except Exception as e:
            print(f" -> ERROR fetching '{table_name}': {e}")
        finally:
            if writer is not None:
                writer.close()

    print("Download complete.")
    return data_dir


# -------------------------
# Download lookup tables
# -------------------------
def _ensure_parent_dir(file_path: str | Path) -> Path:
    """Internal helper to safely create parent directories for an output file."""
    path_obj = Path(file_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj

def export_lookup_counts(engine, table_name: str, columns: list, output_csv: str):
    """Dynamically generate a look table with row counts and saves to CSV"""
    output_path = _ensure_parent_dir(output_csv)

    cols_str = ", ".join(columns)

    query = f"""
    SELECT
        {cols_str},
        COUNT(*) as row_count
    FROM {table_name}
    GROUP BY {cols_str}
    ORDER BY row_count DESC
    """

    df = pd.read_sql(query, engine)
    df.to_csv(output_path, index=False, mode='w', quoting=csv.QUOTE_ALL, na_rep="")
    print(f"Exported {len(df)} distinct pairs to {output_csv}")


def export_lookup_counts_informative(engine, table_name: str,
                                     code_col: str, name_col: str, value_col: str,
                                     output_csv: str, unit_col: str = None):
    """Query to extract look up table, numeric bounds and categorical strings."""
    output_path = _ensure_parent_dir(output_csv)

    unit_select_raw = f"{unit_col}," if unit_col else ""
    unit_select_alias = f"{unit_col} AS unit," if unit_col else "'none' as unit,"
    unit_group = f", {unit_col}" if unit_col else ""

    query = f"""
    WITH parsed_data AS (
    SELECT
        {code_col} as raw_code,
        {name_col} as raw_name,
        {unit_select_raw}
        {value_col} as raw_value,
        TRY_TO_DOUBLE({value_col}) AS numeric_val
    FROM {table_name}
    WHERE {code_col} IS NOT NULL
    )
    SELECT
        raw_code AS code,
        raw_name AS name,
        {unit_select_alias}
        CASE
            WHEN COUNT(numeric_val) > 0 THEN 'float'
            ELSE 'enumerated'
        END AS type,
        MIN(numeric_val) AS range_low,
        MAX(numeric_val) AS range_high,
        LISTAGG(
            DISTINCT CASE WHEN numeric_val IS NULL THEN raw_value END,
            ' | '
        ) AS string_values,
        COUNT(*) AS total_ocurrences
    FROM parsed_data
    GROUP BY
        raw_code,
        raw_name{unit_group}
    ORDER BY
        total_ocurrences DESC;
    """

    df = pd.read_sql(query, engine)
    df.to_csv(output_path, index=False, mode='w', quoting=csv.QUOTE_ALL, na_rep="")
    print(f"Exported {len(df)} distinct pairs to {output_csv}")


def export_multidim_lookup(engine, table_name: str,
                           group_cols: list, output_csv: str,
                           listagg_col: str = None,
                           listagg_alias: str = "string_values") -> pd.DataFrame:
    """Query to extract look up table, numeric bounds and categorical strings."""
    output_path = _ensure_parent_dir(output_csv)

    select_clause = ",\n    ".join(group_cols)
    group_clause = ",\n    ".join(group_cols)

    agg_clause = ""
    if listagg_col:
        agg_clause = f",\n    LISTAGG(DISTINCT {listagg_col}, ' | ') AS {listagg_alias}"

    query = f"""
    SELECT 
        {select_clause}
        {agg_clause},
        COUNT(*) AS total_ocurrences
    FROM {table_name}
    WHERE {group_cols[0]} IS NOT NULL
    GROUP BY 
        {group_clause}
    ORDER BY 
        total_ocurrences DESC;
    """

    df = pd.read_sql(query, engine)
    df.to_csv(output_path, index=False, mode='w', quoting=csv.QUOTE_ALL, na_rep="")
    print(f"Exported {len(df)} distinct pairs to {output_csv}")

