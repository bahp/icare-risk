# Libraries
import pandas as pd
from pathlib import Path
from datetime import datetime

from .fetch import fetch_clinical_records

def download_and_save_tables(tables: list,
                            output_dir: str | Path = './data/raw',
                            **fetch_kwargs):
    """
    Downloads a batch of database tables and saves them to a timestamped
    directory on disk.

    Parameters
    ----------
    tables : list of str
        A list of table names to download from the database.
    output_dir : str or pathlib.Path, optional
        The parent directory where the timestamped folder will be created.
        Default is "./data/synthetic".
    pids : list, optional
        A list of patient identifiers to filter the tables by. Default is None.
    encounter_ids : list, optional
        A list of encounter identifiers to filter the tables by. Default is None.
    file_format : {'csv', 'parquet'}, optional
        The file format used to save the exported tables. Default is 'csv'.
    **fetch_kwargs
        Additional keyword arguments passed directly to `fetch_clinical_records`
        (e.g., `pid_col`, `enc_col`, `backend`).

    Returns
    -------
    pathlib.Path
        The absolute or relative path to the newly created timestamped
        directory containing the saved files.
    """

    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    data_dir = Path(output_dir) / date_str
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting download for {len(tables)} tables...")
    print(f"Saving files to: {data_dir}")

    for table_name in tables:
        print(f"Fetching '{table_name}'...")

        try:
            df = fetch_clinical_records(table_name=table_name, **fetch_kwargs)

            if df.empty:
                print(f"  -> Warning: No records found for '{table_name}'")

            file_path = data_dir / f"{table_name}.csv"
            df.to_csv(file_path, index=False)

            print(f"  -> Saved {len(df)} rows to {file_path}")

        except Exception as e:
            print(f"  -> ERROR fetching '{table_name}': {e}")

    print("Download complete.")
    return data_dir