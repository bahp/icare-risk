"""

Run this test:

    # Ensure your custom folder is discoverable by python
    export PYTHONPATH=.

    # Run just the custom tests
    pytest local/tests/test_pipeline_integration.py -v
"""

import pandas as pd
import pytest

# Adjust this import based on your exact package structure
from icare_risk.features import FeaturePipeline


def test_custom_pipeline_integration():
    """
    Tests that the FeaturePipeline correctly processes base features,
    custom external phenotypes, computed string expressions, and external scores.
    """

    # 1. Mock the configuration (Equivalent to your data_config.yaml)
    mock_config = {
        'base_features': {
            'HR': {
                'missing_indicator': True,
                'impute': 'ffill',
                'delta': True
            }
        },
        'custom_features': {
            'local_stress_flag': {
                'module': 'local.features.stress',
                'function': 'derive_local_stress_phenotype',
                'kwargs': {'hr_col': 'HR', 'temp_col': 'Temp'}
            }
        },
        'computed_features': {
            'HR_to_Temp_Ratio': 'HR / Temp'
        },
        'custom_scores': {
            'local_ward_triage_score': {
                'module': 'local.scores.triage',
                'function': 'calculate_local_ward_triage_score',
                'kwargs': {'age_col': 'AGE_AT_ADMISSION', 'stress_col': 'local_stress_flag'}
            }
        }
    }

    # 2. Create mock static data (Demographics)
    df_static = pd.DataFrame({
        'patient_id': [1, 2],
        'AGE_AT_ADMISSION': [70, 45]  # Patient 1 is older than 65, Patient 2 is younger
    })

    # 3. Create mock time-series data
    # Patient 1: Has high HR (110) & Fever (39.0). We also leave a missing HR to test 'ffill'
    # Patient 2: Normal vitals.
    df_ts = pd.DataFrame({
        'patient_id': [1, 1, 2, 2],
        'date': pd.to_datetime(['2026-08-01', '2026-08-02', '2026-08-01', '2026-08-02']),
        'HR': [110.0, None, 75.0, 80.0],
        'Temp': [39.0, 39.5, 37.0, 37.0]
    })

    # 4. Initialize and run the Pipeline
    pipeline = FeaturePipeline(config_dict=mock_config)
    result_df = pipeline.process(df_static, df_ts)

    # ---------------------------------------------------------
    # 5. Assertions: Verify each of the 4 steps succeeded
    # ---------------------------------------------------------

    # Step 1: Base Features (Missing Indicator & Imputation)
    assert 'HR_is_missing' in result_df.columns
    p1_day2 = result_df[(result_df['patient_id'] == 1) & (result_df['date'] == '2026-08-02')]
    assert p1_day2['HR_is_missing'].iloc[0] == 1  # Successfully flagged the NaN
    assert p1_day2['HR'].iloc[0] == 110.0  # Successfully forward-filled the HR

    # Step 2: Custom Features (Phenotypes)
    assert 'local_stress_flag' in result_df.columns
    p1_day1 = result_df[(result_df['patient_id'] == 1) & (result_df['date'] == '2026-08-01')]
    p2_day1 = result_df[(result_df['patient_id'] == 2) & (result_df['date'] == '2026-08-01')]
    assert p1_day1['local_stress_flag'].iloc[0] == 1  # 110 HR and 39 Temp = Stressed
    assert p2_day1['local_stress_flag'].iloc[0] == 0  # Normal vitals = Not Stressed

    # Step 3: Computed Features (Expressions)
    assert 'HR_to_Temp_Ratio' in result_df.columns
    expected_ratio = 110.0 / 39.0
    assert p1_day1['HR_to_Temp_Ratio'].iloc[0] == expected_ratio

    # Step 4: Custom Scores
    assert 'local_ward_triage_score' in result_df.columns
    # Patient 1: Age > 65 (+2 pts) + Stress Flag (+3 pts) = 5
    assert p1_day1['local_ward_triage_score'].iloc[0] == 5
    # Patient 2: Age < 65 (+0 pts) + No Stress (+0 pts) = 0
    assert p2_day1['local_ward_triage_score'].iloc[0] == 0

    print("\n✅ Pipeline Integration Test Passed!")