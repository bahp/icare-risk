import pandas as pd
from sqlalchemy import MetaData, select

def fetch_records_sa(engine,
                     table_name: str,
                     pids: list = None,
                     encounter_ids: list = None,
                     pid_col='subject',
                     enc_col='encounter_id'):
    """
    Downloads table data filtering by PIDs, encounter IDs, or both
    using SQLAlchemy Core.
    """
    if not pids and not encounter_ids:
        raise ValueError("You must provide at least a list of PIDs or encounter IDs.")

    # 1. Reflect the database metadata
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # 2. Case-insensitive table matching
    actual_table_name = table_name
    if table_name not in metadata.tables and table_name.lower() in metadata.tables:
        actual_table_name = table_name.lower()

    if actual_table_name not in metadata.tables:
        raise ValueError(f"Table '{table_name}' not found in the database.")

    table = metadata.tables[actual_table_name]

    # Helper function for case-insensitive column matching
    def get_column(target_name):
        for col in table.c:
            if col.name.lower() == target_name.lower():
                return col
        return None

    # 3. Build the dynamic WHERE conditions
    conditions = []

    if pids:
        pid_col = get_column(pid_col)
        if pid_col is None:
            raise ValueError(f"Column '{pid_col}' not found in '{actual_table_name}'.")
        # SQLAlchemy handles the IN clause natively
        conditions.append(pid_col.in_(pids))

    if encounter_ids:
        enc_col = get_column(enc_col)
        if enc_col is None:
            raise ValueError(f"Column '{enc_col}' not found in '{actual_table_name}'.")
        conditions.append(enc_col.in_(encounter_ids))

    # 4. Construct the query
    # Unpacking the list with *conditions automatically joins them with AND
    stmt = select(table).where(*conditions)

    # 5. Execute and return as a DataFrame
    with engine.connect() as connection:
        df = pd.read_sql(stmt, connection)
        return df