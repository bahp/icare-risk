# scripts/e_validate_scores.py
import sys
import pandas as pd
import logging
from pathlib import Path

# Setup path to import src dynamically
project_root = Path.cwd()
sys.path.append(str(project_root))

from icare_risk.scores import (
    calculate_increment_esbl,
    calculate_holmgren_score,
    calculate_gavaghan_score,
    calculate_jones_score,
    calculate_tumbarello_score,
    calculate_kim_score
)


def setup_validation_logger(log_path):
    logger = logging.getLogger("ScoreValidator")
    logger.setLevel(logging.INFO)

    # Overwrite the log file each time
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    # Also print to terminal
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    return logger


def main():

    cases_path = project_root / 'tests' / 'cases.csv'
    log_path = project_root / 'reports' / 'score_validation.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_validation_logger(log_path)

    if not cases_path.exists():
        logger.error(f"❌ Error: Validation cases not found at {cases_path}")
        return

    df = pd.read_csv(cases_path, on_bad_lines='skip')

    # 1. Define the Score Dictionary
    # Maps a human-readable name to the function and the expected CSV column
    SCORE_FUNCTIONS = {
        'INCREMENT-ESBL': {'func': calculate_increment_esbl, 'exp_col': 'exp_increment'},
        'Holmgren 2020': {'func': calculate_holmgren_score, 'exp_col': 'exp_holmgren'},
        'Gavaghan 2025': {'func': calculate_gavaghan_score, 'exp_col': 'exp_gavaghan'},
        'Jones 2025': {'func': calculate_jones_score, 'exp_col': 'exp_jones'},
        'Tumbarello': {'func': calculate_tumbarello_score, 'exp_col': 'exp_tumbarello'},
        'Kim 2019': {'func': calculate_kim_score, 'exp_col': 'exp_kim'}
    }

    logger.info("=" * 60)
    logger.info("🏥 CLINICAL SCORE VALIDATION REPORT")
    logger.info("=" * 60 + "\n")

    # 2. Iterate over Cases
    for idx, row in df.iterrows():
        patient_df = pd.DataFrame([row]).reset_index(drop=True)
        desc = row.get('description', f'Case {idx}')

        logger.info(f"▶ CASE ID: {row.get('patient_id', idx)} | {desc}")
        logger.info("=" * 60)

        # 3. Iterate over the Score Dictionary for each case
        for score_name, config in SCORE_FUNCTIONS.items():
            func = config['func']
            exp_col = config['exp_col']

            # Retrieve expected score from CSV, default to None if missing
            expected_score = row.get(exp_col, None)

            # Execute the function (this will trigger the audit_log inside the function)
            calculated_score = func(patient_df, verbose=True, logger=logger).iloc[0]

            # Validation Check
            if expected_score is not None:
                if calculated_score == expected_score:
                    logger.info(f"    ✅ MATCH: Expected {expected_score}, got {calculated_score}\n")
                else:
                    logger.info(f"    ❌ MISMATCH: Expected {expected_score}, got {calculated_score}\n")
            else:
                logger.info(f"    ⚠️ NO EXPECTED SCORE PROVIDED. Computed: {calculated_score}\n")

        logger.info("\n" + "*" * 60 + "\n")


if __name__ == "__main__":
    main()