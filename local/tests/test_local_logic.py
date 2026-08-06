"""

Run this test:

    # Ensure your custom folder is discoverable by python
    export PYTHONPATH=.

    # Run just the custom tests
    pytest local/tests/test_local_logic.py -v
"""

import pandas as pd
import pytest

# Import your locally defined functions
from local.features.stress import derive_local_stress_phenotype
from local.scores.triage import calculate_local_ward_triage_score


# ----------------------------------------------------------------------
# 1. Test the Custom Phenotype
# ----------------------------------------------------------------------
def test_derive_local_stress_phenotype():
    """
    Tests the logic where BOTH tachycardia (>100) and fever (>38.0)
    must be present to flag as 1.
    """
    # Set up 3 dummy patients
    data = pd.DataFrame({
        'HR': [110, 115, 80],  # High, High, Normal
        'Temp': [39.0, 37.0, 36.5]  # High, Normal, Normal
    })

    # Execute the phenotype function
    result = derive_local_stress_phenotype(data, hr_col='HR', temp_col='Temp')

    # Assertions
    assert result[0] == 1  # Patient 0: HR 110 & Temp 39.0 -> Meets both criteria
    assert result[1] == 0  # Patient 1: HR 115 but no fever -> Does not meet criteria
    assert result[2] == 0  # Patient 2: Normal vitals -> Does not meet criteria


# ----------------------------------------------------------------------
# 2. Test the Custom Score
# ----------------------------------------------------------------------
def test_calculate_local_ward_triage_score_max():
    """
    Verify maximum point allocation: Age > 65 (+2 pts) and Stress Flag (+3 pts).
    """
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [70],
        'local_stress_flag': [1]
    })

    score = calculate_local_ward_triage_score(data, age_col='AGE_AT_ADMISSION', stress_col='local_stress_flag')

    assert score[0] == 5, f"Expected MAX 5, got {score[0]}."


def test_calculate_local_ward_triage_score_min():
    """
    Verify minimum point allocation ("Healthy Patient" scenario).
    """
    data = pd.DataFrame({
        'AGE_AT_ADMISSION': [25],  # < 65 years old
        'local_stress_flag': [0]  # No stress phenotype
    })

    score = calculate_local_ward_triage_score(data, age_col='AGE_AT_ADMISSION', stress_col='local_stress_flag')

    assert score[0] == 0, f"Expected MIN 0, got {score[0]}."