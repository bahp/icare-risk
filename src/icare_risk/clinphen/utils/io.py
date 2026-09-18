# Libraries
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# -------------------------
# Utils
# -------------------------
def check_lengths(data_dir: str | Path, file_format: str = 'parquet'):
    """
    Reads files in a directory and prints their total row counts.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        The directory path containing the target files to be evaluated.
    file_format : str, optional
        The format of the files to check, usually 'parquet' or 'csv'.
        Default is 'parquet'.

    Returns
    -------
    None
        This function does not return a value; it prints.
    """
    data_dir = Path(data_dir)
    file_format = file_format.replace('.', '').lower().strip()
    print(f"\n--- Checking {file_format.upper()} File Lengths in {data_dir.name} ---")

    if not data_dir.exists():
        print("Directory does not exist.")
        return

    target_files = list(data_dir.rglob(f'*.{file_format}'))
    if not target_files:
        print(f"No .{file_format} files found in {data_dir}")

    for file_path in target_files:
        try:
            if file_format == 'parquet':
                metadata = pq.read_metadata(file_path)
                row_count = metadata.num_rows
            elif file_format == 'csv':
                row_count = sum(len(chunk) for chunk in pd.read_csv(file_path, chunksize=250_000))
            print(f"{file_path.name:<50} {row_count:>12,} rows")
        except Exception as e:
            print(f"{file_path.name:<50} ERROR reading file ({e})")
    print("-" * 60)