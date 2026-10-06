# Libraries
import os
from pathlib import Path

# So far, these are shared across all datasets that we have created,
# either mock, synthetic or external. They all use the iCare schema.
EPISODE_CONFIG = {
    'subject_col': 'SUBJECT',
    'admission_date': 'ADMISSION_DATE',
    'admission_time': 'ADMISSION_TIME',
    'admission_col': 'std_admission_time',
    'discharge_date': 'DISCHARGE_DATE',
}

TABLE_CONFIG = {
    'vitals': {
        'time_column': 'OBSERVATION_PERFORMED_DT',
        'name_column': 'OBSERVATION_NAME',
        'code_column': 'OBSERVATION_CODE',
        'value_column': 'OBSERVATION_RESULT_CLEAN'
    },
    'pathology': {
        'time_column': 'SAMPLE_COLLECTED_DT',
        'name_column': 'ORDER_NAME',
        'code_column': 'ORDER_CODE',
        'value_column': 'RESULT_CLEANED'
    },
    'microbiology': {
        'time_column': 'LATEST_COLLECT_DT',
        'name_column': 'TEST_NAME',
        'code_column': 'TEST_CODE',
        'value_column': 'ORGANISM_BUG'
    },
    'problems': {
        'time_column': 'PROBLEM_DT_TM',
        'name_column': 'PROBLEM_DESC',
        'code_column': 'PROBLEM_CODE',
        'value_column': None
    }
}


def get_dataset_paths(env: str,
                      dataset_name: str = None,
                      base_dir: str = '/app/data') -> dict:
    """
    Dynamically generates the file paths for the requested environment.
    """
    base_path = Path(base_dir)
    paths = {'table_paths': {}}

    if env == 'mock':
        if not dataset_name:
            raise ValueError("Must provide dataset_name for mock environments (e.g., 'sirs_test').")

        target_dir = base_path / 'mock' / dataset_name
        paths['episodes_path'] = str(target_dir / 'episodes.csv')

        # Dynamically map any CSV found in the mock folder matching our known tables
        for table in TABLE_CONFIG.keys():
            table_file = target_dir / f'{table}.csv'
            if table_file.exists():
                paths['table_paths'][table] = str(table_file)

    elif env == 'synthetic':
        syn_dir = base_path / 'synthetic'

        if dataset_name:
            target_dir = syn_dir / dataset_name
        else:
            # Auto-discover the most recent folder if no specific timestamp is provided
            subdirs = [d for d in syn_dir.iterdir() if d.is_dir()]
            if not subdirs:
                raise FileNotFoundError(f"No synthetic datasets found in {syn_dir}")
            target_dir = max(subdirs, key=os.path.getmtime)

        paths['episodes_path'] = str(target_dir / 'icare_episodes_anon.parquet')

        syn_file_map = {
            'vitals': 'icare_vital_signs_anon.parquet',
            'pathology': 'icare_pathology_blood_anon.parquet',
            'microbiology': 'icare_microbiology_anon.parquet',
            'problems': 'icare_problems_anon.parquet'
        }

        for table, filename in syn_file_map.items():
            file_path = target_dir / filename
            if file_path.exists():
                paths['table_paths'][table] = str(file_path)
    else:
        raise ValueError(f"Unknown environment: {env}")

    return paths