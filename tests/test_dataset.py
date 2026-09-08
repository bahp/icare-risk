# tests/test_dataset.py
import pytest
import pandas as pd
import tempfile
import os
from icare_risk.core.dataset import ClinicalDataset

@pytest.fixture
def dummy_dataset():
    """Creates a temporary dataset with tricky date edges to test the engine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Dummy Episodes: Patient 1 has two stays
        episodes = pd.DataFrame({
            'SUBJECT': [1, 1],
            'ADMISSION_DATE': ['2023-01-10', '2023-05-20'],
            'ADMISSION_TIME': ['10:00:00', '14:00:00'],
            'DISCHARGE_DATE': ['2023-01-15', '2023-05-25']
        })
        episodes_path = os.path.join(tmpdir, 'episodes.csv')
        episodes.to_csv(episodes_path, index=False)

        # 2. Dummy Vitals: Spread across time
        vitals = pd.DataFrame({
            'SUBJECT': [1, 1, 1, 1, 1],
            'OBSERVATION_PERFORMED_DT': [
                '2023-01-09 10:00:00', # History (Day before Stay 1)
                '2023-01-10 09:59:59', # History (Strictly 1 second before Stay 1)
                '2023-01-12 09:00:00', # Current (During Stay 1)
                '2023-03-01 10:00:00', # History for Stay 2 (Happened between stays)
                '2023-05-21 11:00:00'  # Current (During Stay 2)
            ]
        })
        vitals_path = os.path.join(tmpdir, 'vitals.csv')
        vitals.to_csv(vitals_path, index=False)

        # 3. Configure the engine
        episode_config = {
            'admission_date': 'ADMISSION_DATE',
            'admission_time': 'ADMISSION_TIME',
            'discharge_date': 'DISCHARGE_DATE'
        }
        table_config = {'vitals': {'time_column': 'OBSERVATION_PERFORMED_DT'}}

        # Yield gives the dataset to the test, then cleans up the temp files after
        yield ClinicalDataset(
            episodes_path, episode_config, {'vitals': vitals_path}, table_config
        )


def test_temporal_boundaries(dummy_dataset):
    """The Absolute Proof test."""
    
    # 1. Fetch data
    curr = dummy_dataset.get_current_stay('vitals')
    hist = dummy_dataset.get_historical('vitals')

    # 2. Prove Historical logic is perfect
    # The assert statement will crash the test if the condition inside is False
    assert not hist.empty, "Historical data should have been found."
    
    is_strictly_before = (hist['standard_time'] < hist['std_admission_time']).all()
    assert is_strictly_before, "CRITICAL LEAK: Historical data contained events after admission!"

    # 3. Prove Current Stay logic is perfect
    assert not curr.empty, "Current stay data should have been found."
    
    is_after_admission = (curr['standard_time'] >= curr['std_admission_time']).all()
    assert is_after_admission, "CRITICAL LEAK: Current stay data contained events prior to admission!"
    
    is_before_discharge = (curr['standard_time'] <= curr['std_discharge_time']).all()
    assert is_before_discharge, "CRITICAL LEAK: Current stay data contained events after discharge!"