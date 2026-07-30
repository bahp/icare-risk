# tests/test_phenotypes.py
import pandas as pd
import pytest

from icare_risk.phenotypes import (
    derive_pitt_fever_status,
    derive_sirs_tachycardia,
    derive_sirs_tachypnea,
    derive_sirs_abnormal_temp,
    derive_sirs_abnormal_wbc,
    derive_pitt_hypotension_status
)

# ----------------------------------------------------------------------
# Individual tests
# ----------------------------------------------------------------------
def test_derive_pitt_fever_status():
    data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'temp_max': [37.0, 39.5, 40.5],  # 0 pts, 1 pt, 2 pts
        'temp_min': [37.0, 37.0, 34.0]  # Normal, Normal, 2 pts
    })

    # Run the function
    result = derive_pitt_fever_status(data, temp_cols=['temp_max', 'temp_min'])

    assert result.iloc[0] == 0  # Healthy
    assert result.iloc[1] == 1  # Moderate Fever
    assert result.iloc[2] == 2  # Extreme (Takes max of 40.5 and 34.0)


# ----------------------------------------------------------------------
# Fixture related tests
# ----------------------------------------------------------------------
@pytest.fixture
def mock_clinical_data():
    """Provides one sick patient (101) and one healthy patient (102)."""
    return pd.DataFrame({
        'patient_id': [101, 102],
        'hr_24h_max': [110, 75],       # 101: >90 (True)
        'rr_24h_max': [25, 14],        # 101: >20 (True)
        'temp_24h_max': [39.1, 37.0],  # 101: >38 (True)
        'temp_24h_min': [37.0, 36.5],
        'wbc_24h_max': [15.5, 7.2],    # 101: >12 (True)
        'sbp_24h_min': [85, 120]       # 101: <90 (True)
    })

def test_derive_sirs_tachycardia_logic(mock_clinical_data):
    # Function returns a Series
    result = derive_sirs_tachycardia(mock_clinical_data, hr_col='hr_24h_max')
    assert result.iloc[0] == 1  # Patient 101
    assert result.iloc[1] == 0  # Patient 102

def test_derive_sirs_tachypnea_logic(mock_clinical_data):
    result = derive_sirs_tachypnea(mock_clinical_data, rr_col='rr_24h_max')
    assert result.iloc[0] == 1
    assert result.iloc[1] == 0

def test_derive_sirs_abnormal_temp_logic(mock_clinical_data):
    result = derive_sirs_abnormal_temp(mock_clinical_data,
        temp_max_col='temp_24h_max', temp_min_col='temp_24h_min')
    assert result.iloc[0] == 1
    assert result.iloc[1] == 0

def test_derive_sirs_abnormal_wbc_logic(mock_clinical_data):
    result = derive_sirs_abnormal_wbc(mock_clinical_data, wbc_col='wbc_24h_max')
    assert result.iloc[0] == 1
    assert result.iloc[1] == 0

def test_derive_pitt_hypotension_status_logic(mock_clinical_data):
    result = derive_pitt_hypotension_status(mock_clinical_data, sbp_col='sbp_24h_min')
    assert result.iloc[0] == 1
    assert result.iloc[1] == 0

def test_derive_sirs_abnormal_fever_overlap_logic():
    """Ensure bitwise OR is used so result is 1, not 2, for extreme cases."""
    data = pd.DataFrame({
        'temp_24h_max': [40.0],
        'temp_24h_min': [34.0]
    })
    # If using (max > 38) | (min < 36), this should be 1
    result = derive_sirs_abnormal_temp(data,
        temp_max_col='temp_24_max', temp_min_col='temp_24_min')
    assert result.iloc[0] == 1, f"Expected binary 1, got {result.iloc[0]}"