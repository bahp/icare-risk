# icare_risk/utils/debug.py

from pathlib import Path
from typing import Optional
import pandas as pd


def summarize_phenotype_history(
        df_out: pd.DataFrame,
        print_report: bool = True,
        output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Inspects a generated phenotype history dataframe, prints a formatted

    summary report to the terminal, and returns a structured summary dataframe.

    This utility scans the output dataframe for clinical feature columns matching
    the naming conventions (`_last`, `_first`, and `count_prior_`), computes
    basic distribution statistics (valid counts, minimum, mean, maximum), and
    presents a clean terminal visualization.

    Parameters
    ----------
    df_out : pd.DataFrame
        The generated dataframe containing phenotype history features
        (e.g., columns ending with '_last', '_first', or starting with 'count_prior_').
    print_report : bool, default True
        Whether to print the formatted terminal summary report grouped by metric.
    output_csv_path : Optional[Path or str], default None
        Optional file path to export the resulting summary DataFrame as a CSV file.
        Parent directories will be created automatically if they do not exist.

    Returns
    -------
    pd.DataFrame
        A structured summary dataframe where each row corresponds to a evaluated
        phenotype and metric combination, containing the following columns:
        - `phenotype`: Name of the parent clinical phenotype.
        - `metric`: Type of history metric ('last', 'first', or 'count').
        - `column_name`: Full column name in `df_out`.
        - `valid_records`: Number of non-null rows (or records with data).
        - `min`: Minimum value found for the metric.
        - `mean`: Average value across valid records.
        - `max`: Maximum value found for the metric.

    Examples
    --------
    >>> from pathlib import Path
    >>> from icare_risk.utils.debug import summarize_phenotype_history
    >>>
    >>> df_summary = summarize_phenotype_history(
    ...     df_features,
    ...     print_report=True,
    ...     output_csv_path=Path("data/processed/summary_report.csv")
    ... )
    """
    cols = df_out.columns.tolist()
    last_cols = [c for c in cols if c.endswith("_dslast")]
    first_cols = [c for c in cols if c.endswith("_dsfirst")]
    count_cols = [c for c in cols if c.endswith("_cnt")]

    if print_report:
        print(f"\n--- icare_risk Feature Generation Summary ---")
        print(f"Total episodes in dataset: {len(df_out):,}")
        print(f"Metrics found -> Last: {len(last_cols)} | First: {len(first_cols)} | Count: {len(count_cols)}\n")

    summary_records = []

    def _extract_stats(col_list, metric_type):
        for col in col_list:
            if metric_type == "last":
                phenotype = col.replace("_dslast", "")
            elif metric_type == "first":
                phenotype = col.replace("_dsfirst", "")
            else:
                phenotype = col.replace("_cnt", "")

            series = df_out[col]
            valid_count = series.notna().sum()

            if valid_count > 0:
                min_val, max_val, mean_val = series.min(), series.max(), series.mean()
            else:
                min_val, max_val, mean_val = None, None, None

            summary_records.append({
                "phenotype": phenotype,
                "metric": metric_type,
                "column_name": col,
                "valid_records": valid_count,
                "min": min_val,
                "mean": mean_val,
                "max": max_val
            })

    _extract_stats(last_cols, "last")
    _extract_stats(first_cols, "first")
    _extract_stats(count_cols, "count")

    df_summary = pd.DataFrame(summary_records)

    if print_report:
        for metric in ["last", "first", "count"]:
            subset = df_summary[df_summary["metric"] == metric]
            valid_subset = subset[subset["valid_records"] > 0]

            print(f"\n==========================================")
            print(f" METRIC: {metric.upper()} ({len(valid_subset)} active phenotypes)")
            print(f"==========================================")

            if valid_subset.empty:
                print("  (No valid records found)")
                continue

            max_name_len = max(len(p) for p in valid_subset["phenotype"])

            for _, row in valid_subset.iterrows():
                p_name = row["phenotype"]
                v_count = row["valid_records"]

                if metric in ["last", "first"]:
                    print(
                        f"  {p_name:<{max_name_len}} | "
                        f"{v_count:>7,} episodes | "
                        f"Range: {row['min']:>6.1f} to {row['max']:>6.1f} days (Mean: {row['mean']:>6.1f})"
                    )
                else:
                    print(
                        f"  {p_name:<{max_name_len}} | "
                        f"{v_count:>7,} episodes | "
                        f"Range: {row['min']:>2.0f} to {row['max']:>3.0f} events (Mean: {row['mean']:>4.2f})"
                    )

    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_csv_path, index=False)
        if print_report:
            print(f"\nSummary report saved to: {output_csv_path}")

    return df_summary


def print_phenotype_column_matrix(df_out: pd.DataFrame) -> None:
    """Prints a structured checkmark matrix showing which metrics (last, first, count)

    are present and populated with data for each phenotype.
    """
    cols = df_out.columns.tolist()
    last_phenotypes = {c.replace("_dslast", "") for c in cols if c.endswith("_dslast")}
    first_phenotypes = {c.replace("_dsfirst", "") for c in cols if c.endswith("_dsfirst")}
    count_phenotypes = {c.replace("_cnt", "") for c in cols if c.endswith("_cnt")}

    all_phenotypes = sorted(list(last_phenotypes | first_phenotypes | count_phenotypes))

    print(f"\n--- Phenotype Column Matrix ({len(all_phenotypes)} phenotypes found) ---\n")

    if all_phenotypes:
        max_name_len = max(len(p) for p in all_phenotypes)

        for p in all_phenotypes:
            col_last = f"{p}_dslast"
            col_first = f"{p}_dsfirst"
            col_count = f"{p}_cnt"

            has_last = col_last in cols and df_out[col_last].notna().sum() > 0
            has_first = col_first in cols and df_out[col_first].notna().sum() > 0
            has_count = col_count in cols and df_out[col_count].notna().sum() > 0

            status_str = (
                f"[{'✓' if has_last else ' '}] last   "
                f"[{'✓' if has_first else ' '}] first   "
                f"[{'✓' if has_count else ' '}] count"
            )

            print(f"  {p:<{max_name_len}} | {status_str}")
    else:
        print("  (No phenotype history columns found)")
    print()


def print_phenotype_columns_by_metric(df_out: pd.DataFrame) -> None:
    """Prints a categorized, flat list of columns grouped by metric type

    (Recency, Chronicity, Frequency), indicating data validity with checkmarks.
    """
    print(f"\n--- Output Columns Grouped by Metric Type ---\n")

    metric_groups = [
        ("Recency (Last) Columns", [c for c in df_out.columns if c.endswith("_dslast")]),
        ("Chronicity (First) Columns", [c for c in df_out.columns if c.endswith("_dsfirst")]),
        ("Frequency (Count) Columns", [c for c in df_out.columns if c.endswith("_cnt")]),
    ]

    for label, metric_cols in metric_groups:
        valid_count = sum(1 for c in metric_cols if df_out[c].notna().sum() > 0)
        print(f"  {label} ({valid_count}/{len(metric_cols)} active):")

        if not metric_cols:
            print("    (None found)")
        for col in sorted(metric_cols):
            has_data = df_out[col].notna().sum() > 0
            marker = "✓" if has_data else " "
            print(f"    [{marker}] {col}")
        print()