# tests/test_scores.py
import pytest
import pandas as pd
from pathlib import Path
from importlib.resources import files

# Ensure src is in the path
#project_root = Path(__file__).resolve().parent.parent
#sys.path.append(str(project_root))

from icare_risk.utils import load_yaml_config
from icare_risk.scores import (
    calculate_mews,
    calculate_sirs,
    calculate_charlson,
    calculate_increment_esbl,
    calculate_holmgren_score,
    calculate_gavaghan_score,
    calculate_jones_score,
    calculate_tumbarello_score,
    calculate_kim_score,
    calculate_charlson_quan,
    calculate_pitt_score
)


# -----------------------------------------------------------------------------
# Test Setup & Data Loading
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def score_configs():
    """Loads the feature_config.yaml once to share across all tests via smart loader."""
    config = load_yaml_config(config_name="feature_config.yaml")
    return config.get('custom_scores', {})


def get_test_data():
    """Safely loads cases.csv from the packaged data resources."""
    try:
        # Load from installed package resources
        csv_path = files('icare_risk').joinpath('tests/fixtures/cases.csv')
        with csv_path.open('r', encoding='utf-8') as f:
            return pd.read_csv(f)
    except (FileNotFoundError, ModuleNotFoundError):
        # Fallback for direct local repository development
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / 'tests' / 'cases.csv'
        return pd.read_csv(csv_path)


# =============================================================================
# 1. Redefined Clinical Logic Tests (Manual Scenarios)
# =============================================================================
def test_charlson_logic(score_configs):
    """
    Validates Charlson Comorbidity Index weights.
    Logic: Age 75 (+3 points) + Diabetes (1) + CKD (2) = 6
    """
    # Use the 'charlson_score' block from your YAML
    kwargs = score_configs.get('charlson_score', {}).get('kwargs', {})

    # We must use the column names defined in your feature_config.yaml
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [75],
        'hx_diabetes_uncomp': [1],
        'hx_renal_mod_sev': [1]  # This is the standard column for CKD in your pipeline
    })

    # Note: calculate_charlson usually takes (df, comorbidities_dict)
    # We pull the weights (comorbidities) directly from the YAML config
    comorbidities = kwargs.get('comorbidities', {})
    result = calculate_charlson(data, comorbidities=comorbidities)

    # 75yo is 35 years over 40. 35 // 10 = 3 points.
    # If your formula uses (75-30)//10, it's 4. Check your scores.py logic!
    # Based on your comment 'expected 7', we assume 4 (age) + 1 (dm) + 2 (ckd)
    assert result[0] == 6, f"Charlson Expected 6, got {result[0]}"
    print("✅ Charlson Index test passed!")

def test_charlson_quan_metastatic(score_configs):
    """
    Verify that Metastatic Cancer trumps non-metastatic (Scenario B).
    Logic: Age 85 (+4) + Metastatic (6) + CHF (1) = 11.
    The 'Solid Tumor' (2) should be ignored because Metastatic (6) exists.
    """
    # Use the 'charlson_quan_score' block from your YAML
    kwargs = score_configs.get('charlson_quan_score', {}).get('kwargs', {})

    # 85-year-old (+4) + Metastatic (6) + CHF (1) = 11
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [85],
        'hx_cancer_met': [1],    # Metastatic Cancer
        'hx_cancer_solid': [1],  # Localized Solid Tumor (Should be trumped)
        'hx_chf': [1]            # Congestive Heart Failure
    })
    result = calculate_charlson_quan(data, **kwargs)
    assert result[0] == 11, f"Expected 11, got {result[0]}"
    print("✅ Charlson Quan Metastatic hierarchy test passed!")

def test_charlson_quan_hierarchy(score_configs):
    """Verify that Moderate Liver Disease trumps Mild Liver Disease."""
    # Fetch the exact columns expected by the function from the YAML
    kwargs = score_configs.get('charlson_quan_score', {}).get('kwargs', {})

    # 65-year-old (+2) + Mild (1) + Moderate (3) = 5
    # Since Moderate trumps Mild, it should be 2+3=5, NOT 2+1+3=6.
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [65],
        'hx_liver_mild': [1],
        'hx_liver_mod_sev': [1]
    })
    score = calculate_charlson_quan(data, **kwargs)[0]
    assert score == 5, f"Expected 5, got {score}. Hierarchy logic failed."


def test_mews_logic(score_configs):
    """Verify MEWS correctly sums extreme derrangements."""
    kwargs = score_configs.get('mews_score', {}).get('kwargs', {})

    # RR=30 (+3), HR=130 (+3), SBP=70 (+3), Temp=34.0 (+2) => Total 11
    data = pd.DataFrame({
        'rr_24h_max': [30],
        'hr_24h_max': [130],
        'sbp_24h_min': [70],
        'temp_24h_max': [34.0]
    })

    score = calculate_mews(data, **kwargs).iloc[0]
    assert score == 11, f"Expected 11, got {score}. MEWS math is incorrect."


def test_pitt_logic(score_configs):
    """Validates Pitt Bacteremia Score logic."""
    kwargs = score_configs.get('pitt_score', {}).get('kwargs', {})
    # Patient: Temp 34 (+2), Hypotension (+2), Vent (+2), Arrest (+4), Mental Status 1 (+1)
    # Total expected: 11
    data = pd.DataFrame({
        'pitt_fever_status_score': [2],   # 2 pts
        'pitt_hypotension_flag': [1],     # 2 pts
        'pitt_mech_vent_flag': [1],       # 2 pts
        'pitt_cardiac_arrest_flag': [1],  # 4 pts
        'pitt_mental_status_score': [1]   # 1 pts
    })
    score = calculate_pitt_score(data, **kwargs)[0]
    assert score == 11, f"Expected 11, got {score}. Pitt math is incorrect."


def test_sirs_logic(score_configs):
    """Verify SIRS count accumulates correctly."""
    kwargs = score_configs.get('sirs_count', {}).get('kwargs', {})

    data = pd.DataFrame({
        'sirs_tachycardia_flag': [1],
        'sirs_tachypnea_flag': [1],
        'sirs_abnormal_temp_flag': [0],
        'sirs_abnormal_wbc_flag': [1]
    })

    score = calculate_sirs(data, **kwargs)[0]
    assert score == 3, f"Expected 3, got {score}. SIRS math is incorrect."


def test_increment_esbl_logic(score_configs):
    """Validates INCREMENT-ESBL based on 2017 Palacios-Baena weights (Max Points)."""
    # Fetch exact column mappings from YAML
    kwargs = score_configs.get('increment_esbl_score', {}).get('kwargs', {})

    # Data uses the engineered flags and configured names (e.g., AGE_AT_ADMISSION)
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [60],                # +3 points
        'increment_is_non_ecoli_flag': [1],      # +2 points (Klebsiella)
        'increment_bsi_not_urinary_flag': [1],   # +3 points (Respiratory)
        'pitt_score': [7],                       # +3 points
        'sirs_count': [3],                       # +4 points
        'charlson_quan_score': [5],              # +4 points
        'increment_abx_inappropriate_flag': [1]  # +2 points
    })

    # Calculation: 3 + 2 + 3 + 3 + 4 + 4 + 2 = 21
    result = calculate_increment_esbl(data, **kwargs).iloc[0]
    assert result == 21, f"INCREMENT-ESBL Expected 21, got {result}"
    print("✅ INCREMENT-ESBL test passed!")


def test_holmgren_logic(score_configs):
    """Validates Holmgren 2020 score for low-resistance settings."""
    kwargs = score_configs.get('holmgren_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'hx_hosp_abroad_12m': [1, 0],  # Patient 0: Abroad (+1)
        'hx_prev_3gcr_culture': [0, 0],
        'hx_prev_3gcr_rectal_swab': [0, 1]  # Patient 1: Swab (+1)
    })

    results = calculate_holmgren_score(data, **kwargs)
    assert results.iloc[0] == 1
    assert results.iloc[1] == 1
    print("✅ Holmgren Score test passed!")


def test_gavaghan_logic(score_configs):
    """Verify Gavaghan 2025 point allocation."""
    kwargs = score_configs.get('gavaghan_score', {}).get('kwargs', {})

    # Patient: Age 70 (+1), Prior ESBL (+4), Nursing Home (+2) = 7
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [70],
        'hx_prior_esbl_365d': [1],
        'hx_nursing_home_resident': [1],
        'hx_urinary_catheter_present': [0],
        'hx_prior_fc_abx_90d': [0]
    })

    result = calculate_gavaghan_score(data, **kwargs).iloc[0]
    assert result == 7
    print("✅ Gavaghan Score test passed!")


def test_jones_logic(score_configs):
    """Verify Jones 2025 point allocation for non-urinary isolates."""
    kwargs = score_configs.get('jones_score', {}).get('kwargs', {})

    # Patient: Prior ESBL (+5) + Transfer (+1) = 6
    data = pd.DataFrame({
        'hx_prior_esbl_180d': [1],
        'hx_prior_abx_30d': [0],
        'hx_chronic_dialysis': [0],
        'hx_transfer_from_hosp': [1]
    })

    result = calculate_jones_score(data, **kwargs).iloc[0]
    assert result == 6, f"Expected 6, got {result}"
    print("✅ Jones Score test passed!")


def test_kim_logic(score_configs):
    """Verify Kim et al. 2019 point allocation."""
    kwargs = score_configs.get('kim_score', {}).get('kwargs', {})

    # Patient: Prior ESBL (+5) + Nursing Home (+2) = 7
    data = pd.DataFrame({
        'hx_prior_esbl_any': [1],
        'hx_hosp_last_365d': [0],
        'hx_nursing_home_resident': [1],
        'hx_urinary_catheter_present': [0],
        'hx_prior_abx_90d': [0]
    })

    result = calculate_kim_score(data, **kwargs).iloc[0]
    assert result == 7
    print("✅ Kim Score test passed!")




# =============================================================================
# 1. Maximum Point "Worst-Case" Tests
# =============================================================================
# By setting every single risk factor to "True" or to its most extreme value, we
# can mathematically guarantee that the total score equals the theoretical
# maximum. If the function drops even a single point, the assert statement will
# instantly fail and warn you!

def test_charlson_quan_max(score_configs):
    """Verify Charlson Quan achieves absolute maximum score (31 points)."""
    kwargs = score_configs.get('charlson_quan_score', {}).get('kwargs', {})

    # Age >= 80 (4) + all 8 simple conditions (8) +
    # Severe Liver (3) + Complicated Diabetes (2) + Metastatic Cancer (6) +
    # AIDS (6) + Severe Renal (2) = 31 points.
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [85],
        'hx_mi': [1], 'hx_chf': [1], 'hx_pvd': [1], 'hx_stroke': [1],
        'hx_dementia': [1], 'hx_pulmonary': [1], 'hx_rheum': [1], 'hx_pud': [1],
        'hx_liver_mild': [1], 'hx_liver_mod_sev': [1],
        'hx_diabetes_uncomp': [1], 'hx_diabetes_comp': [1],
        'hx_cancer_solid': [1], 'hx_cancer_met': [1],
        'hx_hiv': [1], 'hx_aids': [1],
        'hx_renal_mod_sev': [1]
    })

    score = calculate_charlson_quan(data, **kwargs)[0]
    assert score == 31, f"Expected MAX 31, got {score}. Hierarchy or sum failed."


def test_mews_max(score_configs):
    """Verify MEWS correctly sums extreme derangements to maximum (11 points)."""
    kwargs = score_configs.get('mews_score', {}).get('kwargs', {})

    # RR>=30 (3) + HR>=130 (3) + SBP<=70 (3) + Temp<35 (2) = 11 points
    data = pd.DataFrame({
        'rr_24h_max': [35],
        'hr_24h_max': [140],
        'sbp_24h_min': [60],
        'temp_24h_max': [34.0]
    })

    score = calculate_mews(data, **kwargs).iloc[0]
    assert score == 11, f"Expected MAX 11, got {score}. MEWS math is incorrect."


def test_sirs_max(score_configs):
    """Verify SIRS count hits maximum (4 points)."""
    kwargs = score_configs.get('sirs_count', {}).get('kwargs', {})

    # Tachycardia (1) + Tachypnea (1) + Temp (1) + WBC (1) = 4 points
    data = pd.DataFrame({
        'sirs_tachycardia_flag': [1],
        'sirs_tachypnea_flag': [1],
        'sirs_abnormal_temp_flag': [1],
        'sirs_abnormal_wbc_flag': [1]
    })

    score = calculate_sirs(data, **kwargs)[0]
    assert score == 4, f"Expected MAX 4, got {score}. SIRS math is incorrect."


def test_pitt_max(score_configs):
    """Verify Pitt hits maximum severity (14 points)."""
    kwargs = score_configs.get('pitt_score', {}).get('kwargs', {})

    # Temp (2) + Hypotension (2) + Vent (2) + Arrest (4) + Mental Coma (4) = 14 points
    data = pd.DataFrame({
        'pitt_fever_status_score': [2],
        'pitt_hypotension_flag': [1],
        'pitt_mech_vent_flag': [1],
        'pitt_cardiac_arrest_flag': [1],
        'pitt_mental_status_score': [4]
    })

    score = calculate_pitt_score(data, **kwargs)[0]
    assert score == 14, f"Expected MAX 14, got {score}. Pitt math is incorrect."


def test_increment_esbl_max(score_configs):
    """Verify INCREMENT-ESBL correctly sums to absolute maximum (21 points)."""
    kwargs = score_configs.get('increment_esbl_score', {}).get('kwargs', {})

    # Age>50 (3) + Charlson>3 (4) + Pitt>=6 (3) + SIRS>=2 (4) +
    # Non-Urinary (3) + Non-E.coli (2) + Inappropriate Abx (2) = 21 points
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [60],
        'charlson_quan_score': [5],
        'pitt_score': [7],
        'sirs_count': [3],
        'increment_bsi_not_urinary_flag': [1],
        'increment_is_non_ecoli_flag': [1],
        'increment_abx_inappropriate_flag': [1]
    })

    score = calculate_increment_esbl(data, **kwargs).iloc[0]
    assert score == 21, f"Expected MAX 21, got {score}. INCREMENT math is incorrect."


def test_gavaghan_max(score_configs):
    """Verify Gavaghan correctly hits maximum risk (10 points)."""
    kwargs = score_configs.get('gavaghan_score', {}).get('kwargs', {})

    # Age>=65 (1) + Prior ESBL (4) + Nursing Home (2) + Catheter (1) + Prior Abx (2) = 10
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [70],
        'hx_prior_esbl_365d': [1],
        'hx_nursing_home_resident': [1],
        'hx_urinary_catheter_present': [1],
        'hx_prior_fc_abx_90d': [1]
    })

    score = calculate_gavaghan_score(data, **kwargs).iloc[0]
    assert score == 10, f"Expected MAX 10, got {score}. Gavaghan math is incorrect."


def test_holmgren_max(score_configs):
    """Verify Holmgren scoring hits maximum (3 points)."""
    kwargs = score_configs.get('holmgren_score', {}).get('kwargs', {})

    # Abroad (1) + Prev Culture (1) + Prev Swab (1) = 3
    data = pd.DataFrame({
        'hx_hosp_abroad_12m': [1],
        'hx_prev_3gcr_culture': [1],
        'hx_prev_3gcr_rectal_swab': [1]
    })

    score = calculate_holmgren_score(data, **kwargs).iloc[0]
    assert score == 3, f"Expected MAX 3, got {score}. Holmgren math is incorrect."


def test_jones_max(score_configs):
    """Verify Jones non-urinary risk score hits maximum (10 points)."""
    kwargs = score_configs.get('jones_score', {}).get('kwargs', {})

    # Prior ESBL (5) + Prior Abx (2) + Dialysis (2) + Transfer (1) = 10
    data = pd.DataFrame({
        'hx_prior_esbl_180d': [1],
        'hx_prior_abx_30d': [1],
        'hx_chronic_dialysis': [1],
        'hx_transfer_from_hosp': [1]
    })

    score = calculate_jones_score(data, **kwargs).iloc[0]
    assert score == 10, f"Expected MAX 10, got {score}. Jones math is incorrect."


def test_tumbarello_max(score_configs):
    """Verify Tumbarello scoring hits maximum (9 points)."""
    kwargs = score_configs.get('tumbarello_score', {}).get('kwargs', {})

    # Prior ESBL (4) + Hosp 90d (2) + Abx 90d (2) + Catheter (1) = 9
    data = pd.DataFrame({
        'hx_prior_esbl_any': [1],
        'hx_hosp_last_90d': [1],
        'hx_prior_abx_90d': [1],
        'hx_urinary_catheter_present': [1]
    })

    score = calculate_tumbarello_score(data, **kwargs).iloc[0]
    assert score == 9, f"Expected MAX 9, got {score}. Tumbarello math is incorrect."


def test_kim_max(score_configs):
    """Verify Kim scoring hits maximum (11 points)."""
    kwargs = score_configs.get('kim_score', {}).get('kwargs', {})

    # Prior ESBL (5) + Hosp 1y (2) + Nursing Home (2) + Catheter (1) + Prior Abx (1) = 11
    data = pd.DataFrame({
        'hx_prior_esbl_any': [1],
        'hx_hosp_last_365d': [1],
        'hx_nursing_home_resident': [1],
        'hx_urinary_catheter_present': [1],
        'hx_prior_abx_90d': [1]
    })

    score = calculate_kim_score(data, **kwargs).iloc[0]
    assert score == 11, f"Expected MAX 11, got {score}. Kim math is incorrect."


# =============================================================================
# 2. Minimum Point "Healthy Patient" Tests
# =============================================================================
# By setting every single risk factor to "False" or to its lower value, we
# can mathematically guarantee that the total score equals the theoretical
# minimum. If the function assigns even a single point, the assert statement will
# instantly fail and warn you!
def test_charlson_quan_min(score_configs):
    """Verify Charlson Quan is 0 for a young patient with no comorbidities."""
    kwargs = score_configs.get('charlson_quan_score', {}).get('kwargs', {})

    # Age < 50 (0 pts) and all hx_ flags set to 0
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [25],
        'hx_mi': [0], 'hx_chf': [0], 'hx_pvd': [0], 'hx_stroke': [0],
        'hx_dementia': [0], 'hx_pulmonary': [0], 'hx_rheum': [0], 'hx_pud': [0],
        'hx_liver_mild': [0], 'hx_liver_mod_sev': [0],
        'hx_diabetes_uncomp': [0], 'hx_diabetes_comp': [0],
        'hx_cancer_solid': [0], 'hx_cancer_met': [0],
        'hx_hiv': [0], 'hx_aids': [0],
        'hx_renal_mod_sev': [0]
    })

    score = calculate_charlson_quan(data, **kwargs)[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_mews_min(score_configs):
    """Verify MEWS is 0 for perfectly stable vital signs."""
    kwargs = score_configs.get('mews_score', {}).get('kwargs', {})

    # Normal ranges: RR 15, HR 75, SBP 120, Temp 37.0
    data = pd.DataFrame({
        'rr_24h_max': [20.5],
        'hr_24h_max': [75],
        'sbp_24h_min': [120],
        'temp_24h_max': [37.0]
    })

    score = calculate_mews(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_sirs_min(score_configs):
    """Verify SIRS count is 0 when no criteria are met."""
    kwargs = score_configs.get('sirs_count', {}).get('kwargs', {})

    data = pd.DataFrame({
        'sirs_tachycardia_flag': [0],
        'sirs_tachypnea_flag': [0],
        'sirs_abnormal_temp_flag': [0],
        'sirs_abnormal_wbc_flag': [0]
    })

    score = calculate_sirs(data, **kwargs)[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_pitt_min(score_configs):
    """Verify Pitt Score is 0 for a stable patient."""
    kwargs = score_configs.get('pitt_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'temp_24h_max': [0],  # Normal temp points
        'hypotension_flag': [0],
        'mech_vent_flag': [0],
        'cardiac_arrest_flag': [0],
        'mental_status_score': [0]  # Alert
    })

    score = calculate_pitt_score(data, **kwargs)[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_increment_esbl_min(score_configs):
    """Verify INCREMENT-ESBL is 0 for a young, healthy patient with urinary source."""
    kwargs = score_configs.get('increment_esbl_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [30],
        'charlson_quan_score': [0],
        'pitt_score': [0],
        'sirs_count': [0],
        'increment_bsi_not_urinary_flag': [0],
        'increment_is_non_ecoli_flag': [0],
        'increment_abx_inappropriate_flag': [0]
    })

    score = calculate_increment_esbl(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_gavaghan_min(score_configs):
    """Verify Gavaghan is 0 when no risk factors are present."""
    kwargs = score_configs.get('gavaghan_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [40],
        'hx_prior_esbl_365d': [0],
        'hx_nursing_home_resident': [0],
        'hx_urinary_catheter_present': [0],
        'hx_prior_fc_abx_90d': [0]
    })

    score = calculate_gavaghan_score(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_holmgren_min(score_configs):
    """Verify Holmgren is 0 for no travel or prior positive cultures."""
    kwargs = score_configs.get('holmgren_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'hx_hosp_abroad_12m': [0],
        'hx_prev_3gcr_culture': [0],
        'hx_prev_3gcr_rectal_swab': [0]
    })

    score = calculate_holmgren_score(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_jones_min(score_configs):
    """Verify Jones is 0 for healthy community patients."""
    kwargs = score_configs.get('jones_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'hx_prior_esbl_180d': [0],
        'hx_prior_abx_30d': [0],
        'hx_chronic_dialysis': [0],
        'hx_transfer_from_hosp': [0]
    })

    score = calculate_jones_score(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_tumbarello_min(score_configs):
    """Verify Tumbarello is 0 for patients with no history of ESBL or hospitalization."""
    kwargs = score_configs.get('tumbarello_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'hx_prior_esbl_any': [0],
        'hx_hosp_last_90d': [0],
        'hx_prior_abx_90d': [0],
        'hx_urinary_catheter_present': [0]
    })

    score = calculate_tumbarello_score(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."


def test_kim_min(score_configs):
    """Verify Kim is 0 for low-risk young patients."""
    kwargs = score_configs.get('kim_score', {}).get('kwargs', {})

    data = pd.DataFrame({
        'hx_prior_esbl_any': [0],
        'hx_hosp_last_365d': [0],
        'hx_nursing_home_resident': [0],
        'hx_urinary_catheter_present': [0],
        'hx_prior_abx_90d': [0]
    })

    score = calculate_kim_score(data, **kwargs).iloc[0]
    assert score == 0, f"Expected MIN 0, got {score}."














def run_all_tests():
    print("--- 🧪 Starting Scoring System Validation ---")
    try:
        test_charlson_logic()
        test_pitt_logic()
        test_increment_esbl_logic()
        # Including your existing ones
        data_mews = pd.DataFrame({'RR': [30], 'HR': [130], 'SBP': [70], 'Temp': [34.0]})
        assert calculate_mews(data_mews)[0] == 10
        print("✅ MEWS test passed!")
        print("\n🎉 ALL SCORING TESTS PASSED!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")


# -----------------------------------------------------------------------------
# 2. Automated Validation from CSV
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("index, row", list(get_test_data().iterrows()))
def test_all_scores_from_csv(index, row, score_configs):
    """
    Loops through cases.csv. For each patient, dynamically loads kwargs
    from the YAML and asserts the calculated score matches the expected score.
    """
    patient_df = pd.DataFrame([row]).reset_index(drop=True)

    # Map the YAML key to the python function and the expected CSV column
    SCORE_MAPPING = {
        'increment_esbl_score': (calculate_increment_esbl, 'exp_increment'),
        'holmgren_score': (calculate_holmgren_score, 'exp_holmgren'),
        'gavaghan_score': (calculate_gavaghan_score, 'exp_gavaghan'),
        'jones_score': (calculate_jones_score, 'exp_jones'),
        'tumbarello_score': (calculate_tumbarello_score, 'exp_tumbarello'),
        'kim_score': (calculate_kim_score, 'exp_kim')
    }

    for yaml_key, (func, exp_col) in SCORE_MAPPING.items():
        expected_score = row.get(exp_col)

        # Skip if no expected score was provided for this specific patient
        if pd.isna(expected_score):
            continue

        # Fetch the kwargs dynamically from the YAML
        func_kwargs = score_configs.get(yaml_key, {}).get('kwargs', {})

        # Calculate and Assert
        calculated_score = func(patient_df, **func_kwargs).iloc[0]
        assert calculated_score == expected_score, \
            f"Failed on Patient {row['patient_id']} for {yaml_key}: Expected {expected_score}, got {calculated_score}"


if __name__ == "__main__":
    run_all_tests()