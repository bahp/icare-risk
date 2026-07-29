# scripts/02_build_features_icare.py

import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
project_root = Path.cwd()

from src.features import FeaturePipeline
from src.utils import get_latest_data_dir




class LazyContextDict(dict):
    """
    A smart dictionary that automatically loads CSVs into pandas DataFrames
    only when they are explicitly requested by a feature extractor.
    """
    def __getitem__(self, key):
        value = super().__getitem__(key)
        # If the value is a Path object, load it on demand!
        if isinstance(value, Path):
            if value.exists():
                print(f"📂 [Lazy Load] Reading context table: {key}")
                df = pd.read_csv(value)
                super().__setitem__(key, df) # Cache it so we only read it once
                return df
            else:
                return pd.DataFrame() # Return empty if file is missing
        return value




def prepare_icare_ts(df_vitals, df_labs, data_config):
    """Pivots ICARE tables using names defined in clinical_concepts."""

    vitals_concepts = data_config.get('clinical_concepts', {}).get('vitals', {})
    v_map = {specs['name']: key for key, specs in vitals_concepts.items()}

    labs_concepts = data_config.get('clinical_concepts', {}).get('labs', {})
    l_map = {specs['name']: key for key, specs in labs_concepts.items()}

    # Pivot Vitals
    vits_wide = df_vitals.pivot_table(
        index=['SUBJECT', 'ENCNTR_ID', 'OBSERVATION_PERFORMED_DT'],
        columns='OBSERVATION_NAME',
        values='OBSERVATION_RESULT_CLEAN',
        aggfunc='mean'
    ).reset_index().rename(columns=v_map)
    print(f"  -> Vitals pivoted. Shape: {vits_wide.shape}")

    # Pivot Labs
    labs_wide = df_labs.pivot_table(
        index=['SUBJECT', 'SAMPLE_COLLECTED_DT'],
        columns='TEST_NAME',
        values='RESULT_CLEANED',
        aggfunc='first'
    ).reset_index().rename(columns=l_map)
    print(f"  -> Labs pivoted. Shape: {labs_wide.shape}")

    # Standardize time and ID columns
    vits_wide = vits_wide.rename(columns={'OBSERVATION_PERFORMED_DT': 'date', 'SUBJECT': 'patient_id'})
    labs_wide = labs_wide.rename(columns={'SAMPLE_COLLECTED_DT': 'date', 'SUBJECT': 'patient_id'})

    # Stack them to ensure we don't lose rows due to mismatched timestamps
    #df_ts = pd.concat([vits_wide, labs_wide]).sort_values(['patient_id', 'date'])
    df_ts = pd.merge(vits_wide, labs_wide, on=['patient_id', 'date'], how='outer').sort_values(['patient_id', 'date'])
    print(f"  -> Combined Time-Series Shape: {df_ts.shape}")

    return df_ts


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Feature Engineering: ICARE Edition")
    parser.add_argument('--data-config',
        type=str, default='data_config.yaml', help='Name of the data YAML config file')
    parser.add_argument('--feature-config',
        type=str, default='feature_config.yaml', help='Name of the feature YAML config file')
    args = parser.parse_args()

    print("==================================================")
    print("⚙️  [Step 2] Feature Engineering: ICARE Edition")
    print("==================================================\n")

    latest_data_dir = get_latest_data_dir()
    run_id = latest_data_dir.name
    print(f"📂 Loading data from run: {run_id}")

    # 1. Load Data
    try:
        df_episodes = pd.read_csv(latest_data_dir / 'icare_episodes_anon.csv')
        df_vitals = pd.read_csv(latest_data_dir / 'icare_vital_signs_anon.csv',
            parse_dates=['OBSERVATION_PERFORMED_DT'])
        df_labs = pd.read_csv(latest_data_dir / 'icare_pathology_blood_anon.csv',
            parse_dates=['SAMPLE_COLLECTED_DT'])
        print("✅ Raw ICARE tables loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required CSV files: {e}")
        return

    data_config_path = project_root / 'config' / args.data_config
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # 2. Prepare Time-Series
    df_ts_wide = prepare_icare_ts(df_vitals, df_labs, data_config)

    # 3. Prepare Static
    df_static = df_episodes.rename(columns={'SUBJECT': 'patient_id'})
    print(f"  -> Static (Episodes) rows: {len(df_static)}")


    # 4. Run Pipeline
    """
    context = {
        'microbiology': pd.read_csv(latest_data_dir / 'icare_microbiology_anon.csv'),
        'pharmacy': pd.read_csv(latest_data_dir / 'icare_pharmacy_prescribing_anon.csv'),
        'episodes': df_episodes,
        'problems': pd.read_csv(latest_data_dir / 'icare_problems_anon.csv')
    }
    """

    context_paths = {
        'microbiology': latest_data_dir / 'icare_microbiology_anon.csv',
        'pharmacy': latest_data_dir / 'icare_pharmacy_prescribing_anon.csv',
        'problems': latest_data_dir / 'icare_problems_anon.csv',
        #'episodes': latest_data_dir / 'icare_episodes_anon.csv',
        #'vitals': latest_data_dir / 'icare_vital_signs_anon.csv',
        #'labs': latest_data_dir / 'icare_pathology_blood_anon.csv',
        'episodes': df_episodes
    }

    context = LazyContextDict(context_paths)
    config_path = project_root / 'config' / args.feature_config
    pipeline = FeaturePipeline(config_path=config_path, context_dfs=context)

    print("\n🚀 Starting Feature Pipeline...")
    final_features = pipeline.process(df_static, df_ts_wide)

    # 5. Save with Debugging
    if final_features is None or final_features.empty:
        print("\n❌ FAIL: Pipeline returned an empty DataFrame. No file will be saved.")
        print("Check: Do 'patient_id' values in Vitals/Labs match 'patient_id' in Episodes?")
    else:
        processed_dir = project_root / 'data' / 'processed' / run_id
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_file = processed_dir / 'features_engineered_icare.csv'
        final_features.to_csv(output_file, index=False)

        try:
            display_path = output_file.relative_to(project_root)
        except ValueError:
            display_path = output_file

        print(f"\n✨ SUCCESS! Engineered {final_features.shape[1]} features across {len(final_features)} rows.")
        print(f"📍 Saved to: {display_path}")


if __name__ == "__main__":
    main()