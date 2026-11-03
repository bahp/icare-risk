import duckdb
import pandas as pd

class Dataset:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = duckdb.connect(db_path)

    def load_csv(self, table_name: str, file_path: str):
        """Loads a CSV into a DuckDB view (or table for in-memory speed)."""
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")

    def register_parquet(self, table_name: str, file_path: str):
        """Maps a Parquet file to a logical table name."""
        self._conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{file_path}')")

    def register_csv(self, table_name: str, file_path: str):
        """Maps a CSV file to a logical table name."""
        self._conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")

    def get_episodes(self, table_name: str = "episodes") -> pd.DataFrame:
        """
        Returns the episodes table as a Pandas DataFrame so the researcher
        can filter their cohort natively in Pandas.
        """
        return self._conn.execute(f"SELECT * FROM {table_name}").df()

    def _execute_query(self, query: str, params: list) -> pd.DataFrame:
        """Internal method used ONLY by EpisodeContext for feature pushdowns."""
        return self._conn.execute(query, params).df()