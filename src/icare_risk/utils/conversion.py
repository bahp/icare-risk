from pathlib import Path
from typing import Literal, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PathLike = Union[str, Path]
Conversion = Literal["csv_to_parquet", "parquet_to_csv"]


def convert_large_csv_to_parquet(
        csv_path: Path,
        parquet_path: Path,
        chunksize: int = 100_000
) -> None:
    """Streams a massive CSV into Parquet in chunks with minimal RAM overhead.

    For massive tabular datasets where schema consistency across millions of
    rows is uncertain, passing dtype=str to pd.read_csv() forces Pandas to
    bypass type inference entirely. It reads every field as a string immediately,
    using significantly less CPU, avoiding chunk-to-chunk schema mismatch errors
    in Parquet, and preventing ArrowInvalid exceptions.
    """

    writer = None

    # Read CSV iteratively in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        # Convert pandas DataFrame chunk to PyArrow Table
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        # Initialize ParquetWriter on the first chunk using its schema
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')

        writer.write_table(table)

    if writer:
        writer.close()

def convert(
    path: PathLike,
    conversion: Conversion,
    delete_originals: bool = False,
    safe_string_objects: bool = True
) -> None:
    """
    Convert CSV files to Parquet or Parquet files to CSV.

    Parameters
    ----------
    path:
        Path to a file or directory.
    conversion:
        Conversion type: "csv_to_parquet" or "parquet_to_csv".
    delete_originals:
        If True, delete the source files after successful conversion.
    safe_string_objects:
        If True, casts Pandas 'object' columns (which cause PyArrow mixed-type crashes)
        to Pandas 'string' type while keeping numeric dtypes (int, float) intact and
        preserving null values.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if conversion == "csv_to_parquet":
        source_suffix = ".csv"
        target_suffix = ".parquet"
    elif conversion == "parquet_to_csv":
        source_suffix = ".parquet"
        target_suffix = ".csv"
    else:
        raise ValueError(
            f"Unsupported conversion: {conversion!r}. "
            "Expected 'csv_to_parquet' or 'parquet_to_csv'."
        )

    if path.is_file():
        if path.suffix.lower() != source_suffix:
            raise ValueError(
                f"Expected a {source_suffix} file, got: {path.name}"
            )
        files = [path]

    elif path.is_dir():
        files = list(path.glob(f"*{source_suffix}"))

        if not files:
            print(f"No {source_suffix} files found in {path}")
            return

    else:
        raise ValueError(f"Invalid path: {path}")

    for source in files:
        target = source.with_suffix(target_suffix)

        print(f"Converting: {source.name} -> {target.name}")

        if conversion == "csv_to_parquet":
            # low memory=False prevents dtype chunk mismatches
            df = pd.read_csv(source, low_memory=False)

            if safe_string_objects:
                # Target ONLY object columns (mixed text/numbers like codes or descriptions)
                # 'string' dtype keeps numeric columns untouched and preserves pd.NA
                object_cols = df.select_dtypes(include=["object"]).columns
                df[object_cols] = df[object_cols].astype("string")

            df.to_parquet(target, index=False)
        else:
            pd.read_parquet(source).to_csv(target, index=False)

        if delete_originals:
            source.unlink()

    print(f"Successfully converted {len(files)} file(s).")


if __name__ == "__main__":

    convert("data/", "csv_to_parquet")
    convert("data.csv", "csv_to_parquet")
    convert("data.parquet", "parquet_to_csv")
    convert("data/", "csv_to_parquet")
    convert("data/", "parquet_to_csv")
    convert("data/", "csv_to_parquet", delete_originals=True)
    convert("data/", "csv_to_json") # Not implemented
    convert("data/", "json_to_csv") # Not implemented
