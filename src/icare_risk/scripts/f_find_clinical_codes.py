import sys
import argparse
import pandas as pd
import yaml
import re
from pathlib import Path
from datetime import datetime

# Setup Paths dynamically
project_root = Path.cwd()
sys.path.append(str(project_root))


def search_dataframe(df, search_cols, contains=None, startswith=None, endswith=None, exclude=None):
    """Applies dynamic filters across one or more columns in the dataframe."""

    # Ensure search columns exist in this specific dataframe
    valid_cols = [col for col in search_cols if col in df.columns]
    if not valid_cols:
        print(f"❌ None of the search columns {search_cols} found. Available: {list(df.columns)}")
        return pd.DataFrame()

    # Concatenate text from all valid search columns into one searchable string per row
    temp_series = df[valid_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

    mask = pd.Series(True, index=df.index)

    # 1. Contains ANY of the keywords
    if contains:
        contains_lower = [k.lower() for k in contains]
        contain_mask = temp_series.apply(lambda x: any(k in x for k in contains_lower))
        mask = mask & contain_mask

    # 2. Excludes ANY of the keywords
    if exclude:
        exclude_lower = [k.lower() for k in exclude]
        exclude_mask = temp_series.apply(lambda x: any(k in x for k in exclude_lower))
        mask = mask & ~exclude_mask

    # 3. Starts with specific string
    if startswith:
        mask = mask & temp_series.str.startswith(startswith.lower())

    # 4. Ends with specific string
    if endswith:
        mask = mask & temp_series.str.endswith(endswith.lower())

    return df[mask]


def process_yaml_config(config_path, lookup_dir_str, out_dir_str):
    """Processes multiple files and campaigns from a YAML file, exporting to a datetimed folder."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    lookup_dir = project_root / lookup_dir_str

    # Create datetimed folder (e.g., outputs/code_searches/20260812_145942/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / out_dir_str / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = ["# MULTI-FILE CLINICAL CODE DISCOVERY REPORT"]
    report_lines.append(f"📂 Output Directory: {out_dir}")
    report_lines.append("=" * 50)

    targets = config_data.get('targets', [])
    if not targets:
        print("❌ No 'targets' list found in config YAML.")
        return

    for target in targets:
        file_name = target.get('file')
        code_col = target.get('code_col', 'code')

        # Formatting columns
        search_cols_raw = target.get('search_cols', ['description'])
        search_cols = [search_cols_raw] if isinstance(search_cols_raw, str) else search_cols_raw

        export_cols_raw = target.get('export_cols')
        export_cols = [export_cols_raw] if isinstance(export_cols_raw, str) else export_cols_raw

        file_path = lookup_dir / file_name
        if not file_path.exists():
            report_lines.append(f"\n⚠️ SKIPPING TARGET: {file_name} (File not found in {lookup_dir})")
            continue

        report_lines.append(f"\n" + "=" * 40)
        report_lines.append(f"📁 TARGET FILE: {file_name}")
        report_lines.append(f"   Code Col:    {code_col}")
        report_lines.append(f"   Search Cols: {search_cols}")
        if export_cols:
            report_lines.append(f"   Export Cols: {export_cols}")
        report_lines.append("=" * 40)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            report_lines.append(f"   ❌ Error reading file: {e}")
            continue

        for campaign in target.get('campaigns', []):
            cat = campaign.get('category', 'Unnamed Category')

            contains = campaign.get('include_keywords') or campaign.get('contains')
            exclude = campaign.get('exclude_keywords') or campaign.get('exclude')
            startswith = campaign.get('startswith')
            endswith = campaign.get('endswith')

            matches = search_dataframe(
                df, search_cols=search_cols, contains=contains,
                startswith=startswith, endswith=endswith, exclude=exclude
            )

            report_lines.append(f"\n## CAMPAIGN: {cat}")
            report_lines.append(f"Contains: {contains}")
            report_lines.append(f"Exclude:  {exclude}")
            report_lines.append("-" * 30)

            if matches.empty:
                report_lines.append("  Total Matches: 0")
            else:
                report_lines.append(f"  Total Matches: {len(matches)}\n")

                # Determine which columns to export
                if export_cols:
                    valid_export_cols = [c for c in export_cols if c in matches.columns]
                else:
                    valid_export_cols = list(matches.columns)  # Fallback to all columns

                # Create a safe filename for the campaign CSV
                safe_cat = re.sub(r'[^a-zA-Z0-9]', '_', cat)
                csv_filename = f"{file_name.replace('.csv', '')}_{safe_cat}.csv"
                csv_path = out_dir / csv_filename

                # Export the CSV
                matches[valid_export_cols].to_csv(csv_path, index=False)
                report_lines.append(f"  💾 Exported CSV: {csv_filename}")

                # Still print to console for quick review
                has_code_col = code_col in matches.columns
                for _, row in matches.iterrows():
                    code_display = row[code_col] if has_code_col else "N/A"
                    desc_display = " | ".join([str(row[c]) for c in search_cols if c in matches.columns])
                    report_lines.append(f"  - Code: {code_display} | Match Text: {desc_display}")

                if has_code_col:
                    unique_codes = matches[code_col].unique().tolist()
                    report_lines.append(f"\n  UNIQUE CODES: {unique_codes}")

    # Output to console and save the summary text file
    final_report = "\n".join(report_lines)
    print(final_report)

    summary_path = out_dir / "00_search_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(final_report)
    print(f"\n✅ All results and summary saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Multi-File Clinical Code Discovery Tool")
    parser.add_argument('--config',
        type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--lookup-dir',
        type=str, default='assets/lookups',
        help='Directory containing lookup CSVs (default: assets/lookups)')
    parser.add_argument('--out-dir',
        type=str, default='outputs/code_searches',
        help='Base directory for output reports and CSVs (default: outputs/code_searches)')
    args = parser.parse_args()

    print(f"🔍 Executing Search Engine via Configuration: {args.config}")
    process_yaml_config(args.config, args.lookup_dir, args.out_dir)


if __name__ == "__main__":
    main()