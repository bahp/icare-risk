import duckdb
import pandas as pd
from pathlib import Path

class ClinicalDataset:

    def __init__(self, episodes_path: str,
                       episode_config: dict,
                       table_paths: dict,
                       table_config: dict):
        self.con = duckdb.connect(':memory:')
        self.table_config = table_config
        
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

        self.subject_col = episode_config.get('subject_col')
        self.admission_col = episode_config.get('admission_col')

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

    def get_events(self, table_name: str,
                         names: list = None,
                         codes: list = None,
                         to_numeric: bool = True) -> pd.DataFrame:
        """Fetches records, filters them, and standardizes the value column."""
        df = self.get_current_stay(table_name)
        if df.empty:
            return df

        config = self.table_config.get(table_name, {})
        name_col = config.get('name_column', 'NAME')
        code_col = config.get('code_column', 'CODE')
        val_col = config.get('value_column', 'RESULT')

        if names and name_col in df.columns:
            df = df[df[name_col].isin(names)].copy()
        if codes and code_col in df.columns:
            df = df[df[code_col].isin(codes)].copy()

        if not df.empty and val_col and val_col in df.columns:
            if to_numeric:
                df['value'] = pd.to_numeric(df[val_col], errors='coerce')
            else:
                df['value'] = df[val_col]

        return df

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
                         time_window_hours: int = None,
                         columns: list = None) -> pd.DataFrame:
        """
        Fetches current stay data, optionally restricted to specific columns.
        """
        window_filter = ""
        if time_window_hours is not None:
            window_filter = f"AND t.standard_time <= e.std_admission_time + INTERVAL {time_window_hours} HOUR"

        # Select specific columns if provided, otherwise select all (t.*)
        if columns:
            safe_cols = list(set(columns + ['SUBJECT']))
            col_str = ", ".join([f"t.{c}" for c in safe_cols])
        else:
            col_str = "t.*"

        query = f"""
            SELECT {col_str}, e.std_admission_time
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


class MockClinicalDataset:
    """
    A mock dataset designed to yield a mix of 0s and 1s across the SIRS criteria.
    """

    def __init__(self):
        self.subject_col = 'SUBJECT'
        self.admission_col = 'std_admission_time'

    def get_current_stay(self, table_name):
        if table_name == 'vitals':
            return pd.DataFrame([
                # PATIENT 101: Classic Sepsis (Should score 4/4)
                [101, '2026-09-01', 'Heart Rate', 13472364, 115.0],  # > 90 (Tachycardia: 1)
                [101, '2026-09-01', 'Respiratory Rate', 9096705, 26.0],  # > 20 (Tachypnea: 1)
                [101, '2026-09-01', 'Temperature C', 10933766, 39.2],  # > 38 (Abnormal Temp: 1)

                # PATIENT 102: Perfectly Healthy (Should score 0/4)
                [102, '2026-09-02', 'Heart Rate', 13472364, 75.0],  # Normal
                [102, '2026-09-02', 'Respiratory Rate', 9096705, 14.0],  # Normal
                [102, '2026-09-02', 'Temperature C', 10933766, 37.1],  # Normal

                # PATIENT 103: Hypothermic & Leukopenic (Temp: 1, WBC: 1)
                [103, '2026-09-03', 'Heart Rate', 13472364, 80.0],  # Normal
                [103, '2026-09-03', 'Respiratory Rate', 9096705, 18.0],  # Normal
                [103, '2026-09-03', 'Temperature C', 10933766, 35.1],  # < 36 (Abnormal Temp: 1)

                # PATIENT 104: Hyperventilation & Bandemia (Tachypnea: 1, WBC: 1)
                [104, '2026-09-04', 'Heart Rate', 13472364, 88.0],  # Normal
                [104, '2026-09-04', 'Respiratory Rate', 9096705, 18.0],  # Normal, but see PaCO2 below
                [104, '2026-09-04', 'Temperature C', 10933766, 37.5],  # Normal

                # PATIENT 105: Missing Labs (Tachycardia: 1, Temp: 1, WBC: 0)
                [105, '2026-09-05', 'Heart Rate', 13472364, 105.0],  # > 90 (Tachycardia: 1)
                [105, '2026-09-05', 'Respiratory Rate', 9096705, 16.0],  # Normal
                [105, '2026-09-05', 'Temperature C', 10933766, 38.5]  # > 38 (Abnormal Temp: 1)
            ], columns=['SUBJECT', 'std_admission_time', 'OBSERVATION_NAME', 'OBSERVATION_CODE',
                        'OBSERVATION_RESULT_CLEAN'])

        elif table_name == 'pathology':
            return pd.DataFrame([
                # PATIENT 101: Classic Sepsis
                [101, '2026-09-01', 'White Blood Cells', 'wbc', 18.5],  # > 12 (Abnormal WBC: 1)
                [101, '2026-09-01', 'PaCO2', None, 38.0],  # Normal

                # PATIENT 102: Perfectly Healthy
                [102, '2026-09-02', 'White Blood Cells', 'wbc', 7.0],  # Normal

                # PATIENT 103: Leukopenic
                [103, '2026-09-03', 'White Blood Cells', 'wbc', 2.5],  # < 4.0 (Abnormal WBC: 1)

                # PATIENT 104: Alternative Criteria triggers (PaCO2 and Bands)
                [104, '2026-09-04', 'White Blood Cells', 'wbc', 8.5],  # Normal WBC count
                [104, '2026-09-04', 'Bands', 'bands', 15.0],  # > 10% (Abnormal WBC: 1)
                [104, '2026-09-04', 'PaCO2', None, 28.0]  # < 32 (Tachypnea: 1)

                # PATIENT 105 has no pathology records intentionally to test missing data handling
            ], columns=['SUBJECT', 'std_admission_time', 'ORDER_NAME', 'ORDER_CODE', 'RESULT_CLEANED'])

        return pd.DataFrame()

    def get_history(self, table_name):
        if table_name == 'problems':
            # This is your exact CSV data, joined with a dummy std_admission_time
            # so the grouper knows which admission we are building features for.
            return pd.DataFrame([
                [10001, '2023-01-01', 'J44.9', 'COPD'],
                [10001, '2023-01-01', 'I10', 'Hypertension'],
                [10001, '2023-01-01', 'I25.21', 'Old Myocardial Infarction'],  # Should match I25.2 prefix
                [10001, '2023-01-01', 'E11.9', 'Type 2 diabetes'],

                [10003, '2023-01-01', 'J44.9', 'COPD'],
                [10003, '2023-01-01', 'E11.9', 'Type 2 diabetes'],
                [10003, '2023-01-01', 'I10', 'Hypertension']
                # 10003 has no MI codes
            ], columns=['SUBJECT', 'std_admission_time', 'PROBLEM_CODE', 'PROBLEM_DESC'])
        return pd.DataFrame()