# scripts/b_build_features_icare.py

import argparse
import pandas as pd
from pathlib import Path

# Setup path
project_root = Path.cwd()

from icare_risk.features import FeaturePipeline
from icare_risk.utils import get_latest_data_dir
from icare_risk.utils import load_yaml_config


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

    def get(self, key, default=None):
        """Override get to ensure it triggers our lazy-loading __getitem__ logic."""
        try:
            # Force the use of bracket notation to trigger __getitem__
            return self[key]
        except KeyError:
            return default


def prepare_icare_ts(df_vitals, df_labs, df_static, data_config):
    """Pivots ICARE tables, preserving native vitals encounters and mapping lab timestamps."""

    settings = data_config.get('pipeline_settings', {})
    pid_col = settings.get('patient_col', 'patient_id')
    enc_col = settings.get('encounter_col', 'ENCNTR_ID')
    adm_col = settings.get('admission_date_col', 'ADMISSION_DATE')
    dis_col = settings.get('discharge_date_col', 'DISCHARGE_DATE')
    date_col = settings.get('ts_date_col', 'date')

    v_cfg = settings.get('tables', {}).get('vitals', {})
    l_cfg = settings.get('tables', {}).get('labs', {})

    vitals_concepts = data_config.get('clinical_concepts', {}).get('vitals', {})
    v_map = {specs['name']: key for key, specs in vitals_concepts.items()}

    labs_concepts = data_config.get('clinical_concepts', {}).get('labs', {})
    l_map = {specs['name']: key for key, specs in labs_concepts.items()}

    # 1. Pivot Vitals (Native ENCNTR_ID preserved)
    vits_wide = df_vitals.pivot_table(
        index=[v_cfg.get('subject_col', 'SUBJECT'), v_cfg.get('encounter_col', 'ENCNTR_ID'), v_cfg.get('date_col', 'OBSERVATION_PERFORMED_DT')],
        columns=v_cfg.get('name_col', 'OBSERVATION_NAME'),
        values=v_cfg.get('val_col', 'OBSERVATION_RESULT_CLEAN'),
        aggfunc='mean'
    ).reset_index().rename(columns={
        v_cfg.get('date_col', 'OBSERVATION_PERFORMED_DT'): date_col,
        v_cfg.get('subject_col', 'SUBJECT'): pid_col,
        v_cfg.get('encounter_col', 'ENCNTR_ID'): enc_col
    })
    vits_wide.rename(columns=v_map, inplace=True)
    vits_wide[date_col] = pd.to_datetime(vits_wide[date_col])

    # 2. Pivot Labs (No native ENCNTR_ID)
    labs_wide = df_labs.pivot_table(
        index=[l_cfg.get('subject_col', 'SUBJECT'), l_cfg.get('date_col', 'SAMPLE_COLLECTED_DT')],
        columns=l_cfg.get('name_col', 'TEST_NAME'),
        values=l_cfg.get('val_col', 'RESULT_CLEANED'),
        aggfunc='first'
    ).reset_index().rename(columns={
        l_cfg.get('date_col', 'SAMPLE_COLLECTED_DT'): date_col,
        l_cfg.get('subject_col', 'SUBJECT'): pid_col
    })
    labs_wide.rename(columns=l_map, inplace=True)
    labs_wide[date_col] = pd.to_datetime(labs_wide[date_col])

    # 3. Map Labs to Encounters using Episode Boundaries
    episodes = df_static[[pid_col, enc_col, adm_col, dis_col]].copy()
    episodes[adm_col] = pd.to_datetime(episodes[adm_col])
    episodes[dis_col] = pd.to_datetime(episodes[dis_col])

    labs_merged = pd.merge(labs_wide, episodes, on=pid_col, how='inner')
    labs_mapped = labs_merged[
        (labs_merged[date_col] >= labs_merged[adm_col]) &
        (labs_merged[date_col] <= labs_merged[dis_col])
    ].drop(columns=[adm_col, dis_col])

    # 4. Merge Vitals and Mapped Labs on patient, encounter, and date
    df_ts = pd.merge(
        vits_wide, labs_mapped,
        on=[pid_col, enc_col, date_col],
        how='outer'
    ).sort_values([pid_col, enc_col, date_col])

    print(f"  -> Synchronized Time-Series Shape: {df_ts.shape}")
    return df_ts


def prepare_icare_ts2(df_vitals, df_labs, data_config):
    """Pivots ICARE tables using names defined in clinical_concepts.

    Parameters
    ----------

    Returns
    -------
    """


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
    vits_wide = vits_wide.rename(columns={
        'OBSERVATION_PERFORMED_DT': 'date', 'SUBJECT': 'patient_id'}
    )
    labs_wide = labs_wide.rename(columns={
        'SAMPLE_COLLECTED_DT': 'date', 'SUBJECT': 'patient_id'}
    )

    # Stack them to ensure we don't lose rows due to mismatched timestamps
    #df_ts = pd.concat([vits_wide, labs_wide]).sort_values(['patient_id', 'date'])
    df_ts = pd.merge(vits_wide, labs_wide,
        on=['patient_id', 'date'], how='outer') \
        .sort_values(['patient_id', 'date'])
    print(f"  -> Combined Time-Series Shape: {df_ts.shape}")

    return df_ts


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Feature Engineering: ICARE Edition")
    parser.add_argument('--data-config',
        type=str, default=None, help='Path to data YAML override')
    parser.add_argument('--feature-config',
        type=str, default=None, help='Path to feature YAML override')
    parser.add_argument('--only',
        type=str, default=None, help='Comma-separated list of specific features or scores to compute (e.g., "hx_mi,pitt_score")')
    args = parser.parse_args()

    print("==================================================")
    print("⚙️  [Step 2] Feature Engineering: ICARE Edition")
    print("==================================================\n")

    # 1. Load the Data Config FIRST
    data_config = load_yaml_config(
        config_name="data_config.yaml",
        user_path=args.data_config
    )

    # 2. Pass the loaded config to find the data directory
    latest_data_dir = get_latest_data_dir(loaded_config=data_config)
    run_id = latest_data_dir.name
    print(f"📂 Loading data from run: {run_id}")

    # 3. Load Data
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

    # 2. Prepare Time-Series
    df_ts_wide = prepare_icare_ts(df_vitals, df_labs, df_episodes, data_config)

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

    #context_paths = {
    #    'microbiology': latest_data_dir / 'icare_microbiology_anon.csv',
    #    'pharmacy': latest_data_dir / 'icare_pharmacy_prescribing_anon.csv',
    #    'problems': latest_data_dir / 'icare_problems_anon.csv',
    #    #'episodes': latest_data_dir / 'icare_episodes_anon.csv',
    #    #'vitals': latest_data_dir / 'icare_vital_signs_anon.csv',
    #    #'labs': latest_data_dir / 'icare_pathology_blood_anon.csv',
    #    'episodes': df_episodes
    #}

    # 6. Set up Context for Features dynamically
    context_paths = {}

    # Read the file mappings from the data config
    file_mappings = data_config.get('context_files', {})

    for context_name, file_name in file_mappings.items():
        context_paths[context_name] = latest_data_dir / file_name

    context = LazyContextDict(context_paths)

    # Load feature config
    feature_config = load_yaml_config(
        config_name="feature_config.yaml",
        user_path=args.feature_config
    )

    # Parse the target list if provided
    targets = [t.strip() for t in args.only.split(',')] if args.only else None
    if targets:
        print(f"🎯 Target Isolation Mode Active: Computing ONLY -> {targets}")

    pipeline = FeaturePipeline(
        config_dict=feature_config,
        data_config=data_config,
        context_dfs=context,
        targets=targets)

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