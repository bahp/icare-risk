import duckdb
import pandas as pd
from pathlib import Path

class ClinicalDataset:
    def __init__(self, episodes_path: str, episode_config: dict, table_paths: dict, table_config: dict):
        self.con = duckdb.connect(':memory:')
        
        # --- NEW: Helper to dynamically choose the right DuckDB reader ---
        def get_read_statement(file_path: str) -> str:
            ext = Path(file_path).suffix.lower()
            if ext in ['.parquet', '.pq']:
                return f"read_parquet('{file_path}')"
            elif ext == '.csv':
                # Added max_line_size=10000000 to prevent crashes on unclosed quotes
                # Optional: add ignore_errors=true if you want it to just skip broken synthetic rows
                return f"read_csv_auto('{file_path}', max_line_size=10000000)"
            else:
                raise ValueError(f"Unsupported file format: '{ext}'. Please use .csv or .parquet")
        # ----------------------------------------------------------------

        # 1. Build the Generic Episodes View
        adm_date = episode_config['admission_date']
        adm_time = episode_config['admission_time']
        disch_date = episode_config['discharge_date']
        
        episodes_read = get_read_statement(episodes_path)
        
        episodes_query = f"""
            CREATE VIEW episodes AS 
            SELECT *,
                CAST({adm_date}::VARCHAR || ' ' || {adm_time}::VARCHAR AS TIMESTAMP) AS std_admission_time,
                CAST({disch_date}::VARCHAR || ' 23:59:59' AS TIMESTAMP) AS std_discharge_time
            FROM {episodes_read}
        """
        self.con.execute(episodes_query)
        
        # 2. Build the Generic Clinical Table Views
        for table_name, path in table_paths.items():
            config = table_config.get(table_name)
            if not config:
                continue
                
            time_col = config['time_column']
            table_read = get_read_statement(path)
            
            table_query = f"""
                CREATE VIEW {table_name} AS 
                SELECT *, 
                       CAST({time_col} AS TIMESTAMP) AS standard_time
                FROM {table_read}
            """
            self.con.execute(table_query)

    def get_current_stay_v0(self, table_name: str) -> pd.DataFrame:
        query = f"""
            SELECT t.*, e.std_admission_time
            FROM {table_name} AS t
            INNER JOIN episodes AS e 
                ON t.SUBJECT = e.SUBJECT
            WHERE t.standard_time >= e.std_admission_time 
              AND t.standard_time <= e.std_discharge_time
        """
        return self.con.query(query).df()


    def get_current_stay(self, 
                         table_name: str, 
                         time_window_hours: int = None) -> pd.DataFrame:
        """
        """
        window_filter = ""
        if time_window_hours is not None:
            window_filter = f"AND t.standard_time <= e.std_admission_time + INTERVAL {time_window_hours} HOUR"
            
        query = f"""
            SELECT t.*, e.std_admission_time
            FROM {table_name} AS t
            INNER JOIN episodes AS e 
                ON t.SUBJECT = e.SUBJECT
            WHERE t.standard_time >= e.std_admission_time 
              AND t.standard_time <= e.std_discharge_time
              {window_filter}
        """
        
        return self.con.query(query).df()

    def get_historical(self, table_name: str) -> pd.DataFrame:
        """
        """
        query = f"""
            SELECT t.*, e.std_admission_time
            FROM {table_name} AS t
            INNER JOIN episodes AS e 
                ON t.SUBJECT = e.SUBJECT
            WHERE t.standard_time < e.std_admission_time
        """
        return self.con.query(query).df()