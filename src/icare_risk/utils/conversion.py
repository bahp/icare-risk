from pathlib import Path
from typing import Literal, Union

import pandas as pd


PathLike = Union[str, Path]
Conversion = Literal["csv_to_parquet", "parquet_to_csv"]


def convert(
    path: PathLike,
    conversion: Conversion,
    delete_originals: bool = False,
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
            pd.read_csv(source).to_parquet(target, index=False)
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
