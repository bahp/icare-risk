# Libraries
import pandas as pd

# -------------------------------------------------------
# Helper methods
# -------------------------------------------------------
def _format_sql_tuple(values: list) -> str:
    """Helper to safely format lists for SQL IN clauses."""
    if not values:
        return "()"
    if len(values) == 1:
        return f"({repr(values[0])})"
    return str(tuple(values))



# -------------------------------------------------------
# Queries
# -------------------------------------------------------
def fetch_records_raw(engine,
                      table_name: str,
                      pids: list,
                      encounter_ids: list = None,
                      pid_column_name: str='subject',
                      enc_column_name: str='encntr_id'):
    """
    Downloads table data filtering by PIDs, encounter IDs, or both using raw SQL strings.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        An active SQLAlchemy database engine used to execute the query.
    table_name : str
        The name of the database table to query.
    pids : list
        A list of patient identifiers to filter by.
    encounter_ids : list, optional
        A list of encounter identifiers to filter by. Default is None.
    pid_column_name : str, optional
        The column name representing the patient identifier in the table.
        Default is 'subject'.
    enc_column_name : str, optional
        The column name representing the encounter identifier in the table.
        Default is 'encntr_id'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered clinical records.

    Raises
    ------
    ValueError
        If neither a list of `pids` nor `encounter_ids` is provided, to prevent
        the accidental download of an entire massive table.
    """
    # Prevent download entire massive table.
    if not pids and not encounter_ids:
        raise ValueError("Yoy must provide at least a list of PIDs or encounter IDs.")

    conditions = []
    if pids:
        conditions.append(f"{pid_column_name} IN {_format_sql_tuple(pids)}")
    if encounter_ids:
        conditions.append(f"{enc_column_name} IN {_format_sql_tuple(encounter_ids)}")
    where_clause = " AND ".join(conditions)

    query = f"""
    SELECT *
    FROM {table_name}
    WHERE {where_clause}
    """

    return pd.read_sql(query, con=engine)

def export_lookup_counts(engine, table_name: str, columns: list, output_csv: str):
    """Dynamically generate a look table with row counts and saves to CSV"""
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
    df.to_csv(output_csv, index=False, mode='w')
    print(f"Exported {len(df)} distinct pairs to {output_csv}")


if __name__ == '__main__':

    # Libraries
    import json
    from snowflake.sqlalchemy import URL
    from sqlalchemy import create_engine

    # Create connection
    conn_string = json.load(open('/opt/ich/python-snowflake-defaults.json'))
    conn_string['database'] = "ICHT_PROD"
    conn_string['schema'] = "ICARE_ICHT"
    engine = create_engine(URL(**conn_string))

    # Export lookups
    export_lookup_counts(
        engine=engine, table_name="ICARE_VITAL_SIGNS_ANON",
        columns=["observation_code", "observation_name", "observation_unit"],
        output_csv='../data/lookups/vital_signs_lookup.csv'
    )

    export_lookup_counts(
        engine=engine, table_name="ICARE_PATHOLOGY_BLOOD_ANON",
        columns=["laboratory_department", "test_code", "test_name", "test_result_unit"],
        output_csv='../data/lookups/pathology_blood_tests_lookup.csv'
    )

    export_lookup_counts(
        engine=engine, table_name="ICARE_PATHOLOGY_BLOOD_ANON",
        columns=["laboratory_department", "order_code", "order_name"],
        output_csv='../data/lookups/pathology_blood_panels_lookup.csv'
    )

    export_lookup_counts(
        engine=engine, table_name="ICARE_PROBLEMS_ANON",
        columns=["problem_code", "problem_desc"],
        output_csv='../data/lookups/problems_lookup.csv'
    )
