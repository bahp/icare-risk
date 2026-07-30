# scripts/d_evaluate_thresholds.py

import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Setup path so Python can find 'src' and 'config'
project_root = Path.cwd()
sys.path.append(str(project_root))

from icare_risk.utils import get_latest_processed_file
from icare_risk.utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Stewardship Safety Evaluation")
    parser.add_argument('--threshold-config',
        type=str, default=None, help='Name of the threshold YAML config override')
    args = parser.parse_args()

    print("==================================================")
    print("⚖️  [Step 4] Stewardship Safety Evaluation")
    print("==================================================\n")

    # 1. Load the latest processed features & Extract Run ID
    processed_file = get_latest_processed_file()
    if not processed_file:
        print("❌ Error: No processed features found. Run Step 2 first.")
        return

    run_id = processed_file.parent.name
    final_features = pd.read_csv(processed_file)

    # 2. Setup Dynamic Output Directory
    out_dir = project_root / 'outputs' / run_id
    metrics_dir = out_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Threshold Configurations
    threshold_config = load_yaml_config(
        config_name="threshold_config.yaml",
        user_path=args.threshold_config
    )
    target_label = threshold_config.get('target_label', 'sepsis_case')

    recommended_thresholds = threshold_config.get('thresholds', {})
    stewardship_results = []

    print(f"📂 Saving results to: outputs/{run_id}/")
    print("\nValidating against literature-recommended thresholds...")

    # 4. Evaluate Fixed Thresholds
    for score_name, cutoff in recommended_thresholds.items():
        if score_name in final_features.columns and target_label in final_features.columns:
            valid_data = final_features[[target_label, score_name]].dropna()
            if valid_data.empty: continue

            y_true = valid_data[target_label]
            y_pred_score = valid_data[score_name]

            # Apply fixed literature threshold
            y_pred_binary = (y_pred_score >= cutoff).astype(int)

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            stewardship_results.append({
                'Score_Name': score_name,
                'Recommended_Cutoff': cutoff,
                'Sensitivity': round(sens, 3),
                'Specificity': round(spec, 3),
                'PPV': round(ppv, 3),
                'NPV': round(npv, 3),
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            })

    # 5. Save and Display
    if stewardship_results:
        steward_df = pd.DataFrame(stewardship_results)
        save_path = metrics_dir / 'stewardship_validation.csv'
        steward_df.to_csv(save_path, index=False)

        print("\n--- Stewardship Safety (NPV Focus) ---")
        print(steward_df[['Score_Name', 'Recommended_Cutoff', 'Sensitivity', 'NPV']].to_string(index=False))
        print(f"\n✅ Pipeline Complete! Results saved to: {save_path.relative_to(project_root)}")
    else:
        print(
            "\n⚠ No thresholds evaluated. Check if your score names in threshold_config.yaml match your computed scores.")


if __name__ == "__main__":
    main()