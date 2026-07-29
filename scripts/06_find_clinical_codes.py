import pandas as pd
import yaml
import os
from pathlib import Path

# Setup Paths
project_root = Path(__file__).resolve().parent.parent
project_root = Path.cwd()
MAP_PATH = project_root / 'assets' / 'clinical_mappings' / 'res195-comorbidity-cci-gold.csv'
CONFIG_PATH = project_root / 'config' / 'code_search.yaml'
OUTPUT_PATH = project_root / 'reports' / 'code_search_results.txt'


def find_codes():
    # 1. Load Data
    if not MAP_PATH.exists():
        print(f"❌ Mapping file not found at {MAP_PATH}")
        return

    df = pd.read_csv(MAP_PATH)
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    report_lines = ["# CLINICAL CODE DISCOVERY REPORT\n" + "=" * 40 + "\n"]

    # 2. Process each campaign
    for campaign in config.get('search_campaigns', []):
        cat = campaign['category']
        inc = [k.lower() for k in campaign.get('include_keywords', [])]
        exc = [k.lower() for k in campaign.get('exclude_keywords', [])]
        charlson_filter = campaign.get('target_charlson')

        # Filter by Charlson Category if specified
        temp_df = df.copy()
        if charlson_filter:
            temp_df = temp_df[temp_df['CharlsonCategory'] == charlson_filter]

        # Case-insensitive description search
        desc = temp_df['description'].fillna('').str.lower()

        # Logic: Must contain ANY include keyword
        # .astype(bool) forces the Series to be a proper boolean mask
        include_mask = desc.apply(lambda x: any(k in x for k in inc)).astype(bool)

        # Logic: Must NOT contain ANY exclude keyword
        exclude_mask = desc.apply(lambda x: any(k in x for k in exc)).astype(bool)

        # Now '&' and '~' (NOT) will work correctly
        final_matches = temp_df[include_mask & ~exclude_mask]

        # 3. Format Report
        report_lines.append(f"\n## CATEGORY: {cat}")
        report_lines.append(f"Include: {inc} | Exclude: {exc}")
        report_lines.append("-" * 30)

        if final_matches.empty:
            report_lines.append("  No matches found.")
        else:
            for _, row in final_matches.iterrows():
                # Extract code and description for your other YAML configs
                report_lines.append(f"  - Code: {row['code']} | Desc: {row['description']}")

            # Summary for easy copy-pasting
            unique_codes = final_matches['code'].unique().tolist()
            report_lines.append(f"\n  UNIQUE CODES FOR YAML: {unique_codes}")

    # 4. Save results
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"✅ Discovery complete. Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    find_codes()