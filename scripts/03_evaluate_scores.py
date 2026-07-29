# scripts/03_evaluate_scores.py

import sys
import yaml
import argparse
import operator
import pandas as pd
from pathlib import Path

# Setup path so Python can find 'src' and 'config'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
project_root = Path.cwd()

from src.metrics import ClinicalEvaluator
from src.utils import get_latest_processed_file


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Configuration-Driven Model Evaluation")
    parser.add_argument('--eval-config',
        type=str, default='eval_config.yaml', help='Name of the evaluation YAML config file')
    parser.add_argument('--feature-config',
        type=str, default='feature_config.yaml', help='Name of the feature YAML config file')
    args = parser.parse_args()

    print("==================================================")
    print("📊 [Step 3] Configuration-Driven Model Evaluation")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. Load Data & Configs
    # ---------------------------------------------------------
    processed_file = get_latest_processed_file()
    final_features = pd.read_csv(processed_file)

    run_id = processed_file.parent.name
    out_dir = project_root / 'outputs' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'run_metadata.txt', 'w') as f:
        f.write(f"--- CLINICAL PIPELINE METADATA ---\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Source Data: {processed_file}\n")

    config_path = project_root / 'config' / 'eval_config.yaml'
    with open(config_path, 'r') as file:
        eval_config = yaml.safe_load(file)

    exp_config = eval_config.get('experiment', {})
    target_label = exp_config.get('target_label', 'ground_truth')
    requested_metrics = exp_config.get('metrics_to_compute', 'all')

    # ---------------------------------------------------------
    # 2. Determine which scores to evaluate
    # ---------------------------------------------------------
    if exp_config.get('scores_to_evaluate') == 'all':
        feature_config_path = project_root / 'config' / args.feature_config
        with open(feature_config_path, 'r') as file:
            fc = yaml.safe_load(file)
            custom_scores = list(fc.get('custom_scores', {}).keys())
            # Exclude the target label to prevent 2D array crashes
            valid_scores = [s for s in custom_scores
                if s in final_features.columns and s != target_label]
    else:
        valid_scores = [s for s in exp_config.get('scores_to_evaluate', [])
            if s in final_features.columns]

    if not valid_scores:
        print("⚠ No valid scores found in the dataset to evaluate.")
        return

    print(f"🎯 Target Label: '{target_label}'")
    print(f"🧪 Evaluating Scores: {', '.join(valid_scores)}")

    # ---------------------------------------------------------
    # 3. Setup Time Math (Hours from Admission)
    # ---------------------------------------------------------
    final_features['date'] = pd.to_datetime(final_features['date'])

    if 'ADMISSION_DATE' not in final_features.columns:
        print("⚠ 'ADMISSION_DATE' missing. Approximating from first recorded clinical event.")
        admin_dates = final_features.groupby('patient_id')['date'].min().reset_index()
        admin_dates.rename(columns={'date': 'ADMISSION_DATE'}, inplace=True)
        final_features = final_features.merge(admin_dates, on='patient_id', how='left')
    else:
        final_features['ADMISSION_DATE'] = pd.to_datetime(final_features['ADMISSION_DATE'])

    # Calculate exact elapsed hours for every row
    final_features['hours_from_admin'] = (final_features['date'] - final_features[
        'ADMISSION_DATE']).dt.total_seconds() / 3600

    evaluator = ClinicalEvaluator(output_dir=out_dir)
    master_results = []

    # ---------------------------------------------------------
    # 4. Prepare Subgroups (Multi-Filter Engine)
    # ---------------------------------------------------------
    subgroup_config = eval_config.get('subgroup_analysis', {})
    cohorts_to_run = [{'name': 'All Patients', 'df': final_features}]

    # Map string operators from YAML to actual Python math operations
    ops = {
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le
    }

    if subgroup_config.get('run', False):
        for cohort in subgroup_config.get('cohorts', []):
            cohort_name = cohort['name']
            master_mask = pd.Series(True, index=final_features.index)
            valid_cohort = True

            for f in cohort.get('filters', []):
                col = f['column']
                op_str = f.get('operator', '==')
                val = f['value']

                if col not in final_features.columns:
                    print(f"⚠️ Skipping '{cohort_name}': Column '{col}' not found in data.")
                    valid_cohort = False
                    break

                if op_str not in ops:
                    print(f"⚠️ Skipping '{cohort_name}': Unknown operator '{op_str}'.")
                    valid_cohort = False
                    break

                op_func = ops[op_str]
                current_mask = op_func(final_features[col], val)
                master_mask = master_mask & current_mask

            if valid_cohort:
                df_sub = final_features[master_mask].copy()
                cohorts_to_run.append({'name': cohort_name, 'df': df_sub})

    # ---------------------------------------------------------
    # 5. Process Time-Slice Experiments (For Every Cohort)
    # ---------------------------------------------------------
    time_slices = eval_config.get('time_slices', {})

    for cohort in cohorts_to_run:
        cohort_name = cohort['name']
        cohort_df = cohort['df']

        if cohort_df.empty:
            print(f"\n⚠️ Skipping {cohort_name}: No patients found matching criteria.")
            continue

        print(f"\n\n🏥 === RUNNING COHORT: {cohort_name.upper()} ({len(cohort_df.patient_id.unique())} Patients) ===")

        if time_slices.get('continuous', {}).get('run'):
            print(f"  -> [Continuous] Evaluating all rows...")
            strategy = f"{cohort_name} - Continuous"
            res = evaluator.evaluate_and_plot(cohort_df,
                                              target_label,
                                              valid_scores,
                                              strategy,
                                              time_slices['continuous'],
                                              requested_metrics)
            master_results.append(res)

        if time_slices.get('admission', {}).get('run'):
            print(f"  -> [Admission] Evaluating first clinical record...")
            strategy = f"{cohort_name} - Admission"
            df_slice = cohort_df \
                .sort_values(['patient_id', 'date']) \
                .groupby('patient_id').first() \
                .reset_index()
            res = evaluator.evaluate_and_plot(df_slice,
                                              target_label,
                                              valid_scores,
                                              strategy,
                                              time_slices['admission'],
                                              requested_metrics)
            master_results.append(res)

        if time_slices.get('peak', {}).get('run'):
            print(f"  -> [Peak Severity] Evaluating worst recorded score...")
            strategy = f"{cohort_name} - Peak Severity"
            df_slice = cohort_df \
                .groupby('patient_id')[valid_scores + [target_label]] \
                .max().reset_index()
            res = evaluator.evaluate_and_plot(df_slice,
                                              target_label,
                                              valid_scores,
                                              strategy,
                                              time_slices['peak'],
                                              requested_metrics)
            master_results.append(res)

        if time_slices.get('milestones', {}).get('run'):
            settings = time_slices['milestones']
            tol = settings.get('tolerance_hours', 4)
            for h in settings.get('hours', []):
                print(f"  -> [Milestone {h}h] Evaluating records near hour {h}...")
                df_window = cohort_df[(cohort_df['hours_from_admin'] >= (h - tol)) &
                                      (cohort_df['hours_from_admin'] <= (h + tol))]
                df_slice = df_window \
                    .sort_values(['patient_id', 'hours_from_admin']) \
                    .groupby('patient_id').first() \
                    .reset_index()

                strategy = f"{cohort_name} - Milestone {h}h"
                res = evaluator.evaluate_and_plot(df_slice,
                                                  target_label,
                                                  valid_scores,
                                                  strategy,
                                                  settings,
                                                  requested_metrics)
                master_results.append(res)

    # ---------------------------------------------------------
    # 6. Process Longitudinal Analyses (Over the whole dataset)
    # ---------------------------------------------------------
    longitudinal = eval_config.get('longitudinal', {})
    for plot_name, settings in longitudinal.items():
        if settings.get('run'):
            evaluator.plot_longitudinal(
                final_features, target_label, valid_scores, plot_name, settings
            )

    # ---------------------------------------------------------
    # 7. Master Summary Output
    # ---------------------------------------------------------
    if master_results:
        final_summary = pd.concat([df for df in master_results
            if not df.empty], ignore_index=True)
        print("\n==================================================")
        print("🏆 MASTER EXPERIMENT SUMMARY")
        print("==================================================")
        print(final_summary.to_string(index=False))

        # Save master summary
        final_summary.to_csv(evaluator.metrics_dir / 'master_summary.csv', index=False)
        print(f"\n✅ All metrics and plots saved to: {evaluator.output_dir.relative_to(project_root)}")
    else:
        print("\n⚠ No evaluation strategies were successfully executed.")


if __name__ == "__main__":
    main()