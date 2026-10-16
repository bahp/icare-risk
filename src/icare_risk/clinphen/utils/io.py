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

def save_phenotype_artifacts(
    df_raw: pd.DataFrame,
    df_windowed: pd.DataFrame,
    df_aggregated: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "icare_",
) -> None:
    """Saves raw events, window-filtered events, and aggregated phenotype dataframes

    to Parquet format with explicit save location messages.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw extracted matching events from `run_vectorized`.
    df_windowed : pd.DataFrame
        Window-filtered events from `evaluate_temporal_windows`.
    df_aggregated : pd.DataFrame
        Aggregated patient/spell level phenotypes from `aggregate_composite_phenotypes`.
    output_dir : Path | str
        Directory where parquet files will be written.
    artifact_prefix : str
        Prefix for the generated filenames (e.g., 'icare', 'esbl_cohort_v1').
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts = {
        f"{prefix}events_extracted_raw.parquet": (
            df_raw,
            "Raw extracted matching events",
        ),
        f"{prefix}events_extracted_windowed.parquet": (
            df_windowed,
            "Window-filtered events",
        ),
        f"{prefix}phenotypes_aggregated.parquet": (
            df_aggregated,
            "Aggregated patient/spell phenotypes",
        ),
    }

    print("\n--- Saving Phenotype Artifacts ---")
    for filename, (df, description) in artifacts.items():
        file_filepath = out_path / filename
        df.to_parquet(file_filepath, index=False)
        print(f" Saved {description} ({len(df):,} rows)")
        print(f"   └─ Path: {file_filepath.resolve()}")
    print("-----------------------------------\n")