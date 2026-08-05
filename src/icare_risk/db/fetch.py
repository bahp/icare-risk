from ._raw_sql import fetch_records_raw
from ._sqlalchemy_core import fetch_records_sa


def fetch_clinical_records(
        engine,
        table_name: str,
        pids: list = None,
        encounter_ids: list = None,
        pid_col: str = 'subject',
        enc_col: str = 'encounter_id',
        backend: str = 'sqlalchemy'
):
    """
    Downloads clinical records from a specified database table filtering by PIDs,
    encounter IDs, or both.

    Parameters
    ----------
    table_name : str
        The name of the database table to query.
    pids : list, optional
        A list of patient identifiers to filter by. Default is None.
    encounter_ids : list, optional
        A list of encounter identifiers to filter by. Default is None.
    pid_col : str, optional
        The column name representing the patient identifier in the table.
        Default is 'subject'.
    enc_col : str, optional
        The column name representing the encounter identifier in the table.
        Default is 'encounter_id'.
    backend : {'sqlalchemy', 'sql'}, optional
        The execution backend to use for the query. Default is 'sqlalchemy'.
    engine : sqlalchemy.engine.Engine, optional
        An active SQLAlchemy database engine. If None, the global engine
        managed by `connection.py` is utilized. Default is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered clinical records. Returns an
        empty DataFrame if no records match or if the table is empty.

    Raises
    ------
    ValueError
        If neither `pids` nor `encounter_ids` are provided, or if an
        invalid `backend` string is supplied.
    """
    #if engine is None:
    #    engine = get_engine()

    if backend == 'sqlalchemy':
        return fetch_records_sa(engine=engine,
            table_name=table_name, pids=pids, encounter_ids=encounter_ids,
            pid_column_name=pid_col, enc_column_name=enc_col)

    elif backend == 'sql':
        return fetch_records_raw(engine,
            table_name, pids, encounter_ids, pid_col, enc_col)

    else:
        raise ValueError("Invalid backend. Choose 'sqlalchemy' or 'sql'.")