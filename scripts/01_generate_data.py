# scripts/01_generate_data.py

import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Setup path so Python can find 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
project_root = Path.cwd()

from src.generators import generate_custom_table, generate_eav_timeseries


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Dynamic Synthetic Data Generation")
    parser.add_argument(
        '--config',
        type=str,
        default='data_config.yaml',  # Fallback if no parameter is passed
        help='Name of the YAML config file located in the config/ directory'
    )
    args = parser.parse_args()


    print("==================================================")
    print("🧬 [Step 1] Dynamic Synthetic Data Generation")
    print("==================================================\n")

    # 1. Setup Paths and Load Config
    config_dir = project_root / 'config'
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    data_dir = project_root / 'data' / 'synthetic' / date_str
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 TARGET SAVE DIRECTORY: {data_dir.absolute()}\n")

    config_path = config_dir / args.config
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # 2. Establish Global Parameters
    params = data_config.get('generation_params', {})
    n_patients = params.get('n_patients', 100)

    # Using 10001 as a clean starting ID for patients
    unique_ids = list(range(10001, 10001 + n_patients))
    primary_key_col = "SUBJECT"

    custom_tables = {}

    # 3. Generate Configured Tables
    if 'tables' in data_config:
        print(f"Generating configured tables for {n_patients} patients...")

        # It's important the YAML is ordered with Parent tables (like Episodes)
        # before Child tables (like Pharmacy or Vitals) so foreign keys exist when needed.
        for table_name, table_config in data_config['tables'].items():

            if not isinstance(table_config, dict) or 'type' not in table_config:
                print(
                    f"  ⚠️ Skipping '{table_name}': Missing 'type' in YAML (needs type: 'relational' or 'eav_timeseries').")
                continue

            print(f" -> Building {table_name} [{table_config['type']}]...")

            if table_config['type'] == 'relational':
                df_table = generate_custom_table(
                    unique_ids=unique_ids,
                    table_config=table_config,
                    custom_tables=custom_tables
                )

            elif table_config['type'] == 'eav_timeseries':
                df_table = generate_eav_timeseries(
                    unique_ids=unique_ids,
                    custom_tables=custom_tables,
                    table_config=table_config,
                    clinical_concepts=data_config,
                    freq=params.get('freq', '4h'),
                    days=params.get('days', 10),
                    primary_key_col=primary_key_col
                )

            # Store in memory for foreign key linking by subsequent tables
            custom_tables[table_name] = df_table

            # Save output to CSV
            file_name = f"{table_name.lower()}.csv"
            df_table.to_csv(data_dir / file_name, index=False)

    try:
        display_path = data_dir.relative_to(project_root)
    except ValueError:
        display_path = data_dir

    print(f"\n✅ Success! All data saved dynamically to: {display_path}")

if __name__ == "__main__":
    main()