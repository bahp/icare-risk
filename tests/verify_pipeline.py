import pandas as pd
import yaml
import sys
from pathlib import Path

# 1. Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import FeaturePipeline


def prepare_icare_ts(df_vitals, df_labs, data_config):
    """
    Copy-pasted logic from your build script to ensure the test
    uses the EXACT same preprocessing as the real production run.
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

    # Pivot Labs
    labs_wide = df_labs.pivot_table(
        index=['SUBJECT', 'SAMPLE_COLLECTED_DT'],
        columns='TEST_NAME',
        values='RESULT_CLEANED',
        aggfunc='first'
    ).reset_index().rename(columns=l_map)

    # Standardize names to 'patient_id' and 'date'
    vits_wide = vits_wide.rename(columns={'OBSERVATION_PERFORMED_DT': 'date', 'SUBJECT': 'patient_id'})
    labs_wide = labs_wide.rename(columns={'SAMPLE_COLLECTED_DT': 'date', 'SUBJECT': 'patient_id'})

    # Combine
    df_ts = pd.merge(vits_wide, labs_wide, on=['patient_id', 'date'], how='outer').sort_values(['patient_id', 'date'])
    return df_ts


def test_computation():
    print("📂 Loading Test Fixtures...")
    fixture_path = project_root / 'tests' / 'fixtures'

    # Load raw data
    df_vitals = pd.read_csv(fixture_path / 'icare_vital_signs_anon.csv', parse_dates=['OBSERVATION_PERFORMED_DT'])
    df_labs = pd.read_csv(fixture_path / 'icare_pathology_blood_anon.csv', parse_dates=['SAMPLE_COLLECTED_DT'])
    df_episodes = pd.read_csv(fixture_path / 'icare_episodes_anon.csv')

    # Load configs
    with open(project_root / 'config' / 'data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # 2. Step 1: Prepare Time-Series (Pivoting)
    print("🛠️ Pivoting vitals and labs...")
    df_ts_wide = prepare_icare_ts(df_vitals, df_labs, data_config)

    # 3. Step 2: Prepare Static (Renaming to match your process logic)
    df_static = df_episodes.rename(columns={'SUBJECT': 'patient_id'})

    # 4. Build Context for Phenotypes
    context = {
        'microbiology': pd.read_csv(fixture_path / 'icare_microbiology_anon.csv'),
        'pharmacy': pd.read_csv(fixture_path / 'icare_pharmacy_prescribing_anon.csv'),
        'episodes': df_episodes,
        'problems': pd.read_csv(fixture_path / 'icare_problems_anon.csv')
    }

    # 5. Initialize and Run Pipeline
    config_path = project_root / 'config' / 'feature_config.yaml'
    pipeline = FeaturePipeline(config_path=config_path, context_dfs=context)

    print("\n🚀 Starting Feature Pipeline Process...")
    # This calls your specific: def process(self, df_static, df_ts)
    final_features = pipeline.process(df_static, df_ts_wide)

    # 6. Save Output to fixtures/outputs
    output_dir = fixture_path / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'test_results_check.csv'
    final_features.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")

    # 7. Verification Assertions
    print("\n🧪 Verifying results...")
    # Note: Use 'patient_id' because you renamed it in step 3
    p002 = final_features[final_features['patient_id'] == 'PATIENT_002'].iloc[0]

    print(f"--- PATIENT_002 Check ---")
    print(f"  Charlson Score: {p002.get('charlson_quan_score')} (Expected: 7)")
    print(f"  ESBL Score: {p002.get('increment_esbl_score')} (Expected: 11)")

    assert p002['charlson_quan_score'] == 7, "❌ Charlson score mismatch!"
    print("✅ Logic verification complete.")


if __name__ == "__main__":
    test_computation()