# src/metrics.py
import re
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ClinicalEvaluator:
    def __init__(self, output_dir='../outputs'):
        """Initializes the evaluator and sets up the output directories."""
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def evaluate_and_plot(self, df, true_label_col, score_cols, strategy_name, settings, requested_metrics):
        """Master method to compute tabular metrics and trigger requested plots."""
        # 1. Compute tabular metrics
        results_df = self._compute_scores(df, true_label_col, score_cols, requested_metrics)

        if results_df.empty:
            return results_df

        # Add strategy name for the master summary
        results_df.insert(0, 'Strategy', strategy_name)

        # Save specific CSV for this strategy
        # Keeps only letters and numbers, replacing everything else with an underscore
        safe_name = re.sub(r'[^a-z0-9]', '_', strategy_name.lower())
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')  # Cleans up double underscores
        results_df.to_csv(self.metrics_dir / f'{safe_name}_metrics.csv', index=False)

        # 2. Generate Requested Plots
        plots = settings.get('plots_to_generate', [])
        if 'roc_curve' in plots:
            self._plot_roc_curves(df, true_label_col, score_cols, f'{safe_name}_roc.png', strategy_name)
        if 'pr_curve' in plots:
            self._plot_pr_curves(df, true_label_col, score_cols, f'{safe_name}_pr.png', strategy_name)

        return results_df

    def _compute_scores(self, df, true_label_col, score_cols, requested_metrics):
        """Evaluates clinical scores and filters output based on requested metrics."""
        results = []
        all_metrics_flag = (requested_metrics == 'all')

        for score in score_cols:
            valid_data = df[[true_label_col, score]].dropna()
            if valid_data.empty: continue

            y_true = valid_data[true_label_col]
            y_pred_score = valid_data[score]

            # Check if we have at least 1 positive and 1 negative case
            if len(np.unique(y_true)) < 2:
                continue

            # Core calculations
            auroc = roc_auc_score(y_true, y_pred_score)
            auprc = average_precision_score(y_true, y_pred_score)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_score)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            y_pred_binary = (y_pred_score >= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            f1 = f1_score(y_true, y_pred_binary)
            youden_j = sens + spec - 1

            # Build row dynamically
            row = {'Score_Name': score}
            if all_metrics_flag or 'auroc' in requested_metrics:
                row['AUROC'] = round(auroc, 3)
            if all_metrics_flag or 'auprc' in requested_metrics:
                row['AUPRC'] = round(auprc, 3)
            if all_metrics_flag or 'optimal_cutoff' in requested_metrics:
                row['Optimal_Cutoff'] = round(
                optimal_threshold, 2)
            if all_metrics_flag or 'sensitivity' in requested_metrics:
                row['Sensitivity'] = round(sens, 3)
            if all_metrics_flag or 'specificity' in requested_metrics:
                row['Specificity'] = round(spec, 3)
            if all_metrics_flag or 'ppv' in requested_metrics:
                row['PPV'] = round(ppv, 3)
            if all_metrics_flag or 'npv' in requested_metrics:
                row['NPV'] = round(npv, 3)
            if all_metrics_flag or 'confusion_matrix' in requested_metrics:
                row['TP'] = int(tp)
                row['TN'] = int(tn)
                row['FP'] = int(fp)
                row['FN'] = int(fn)
            if all_metrics_flag or 'f1' in requested_metrics:
                row['F1_Score'] = round(f1, 3)
            if all_metrics_flag or 'youden' in requested_metrics:
                row['Youden_J'] = round(youden_j, 3)

            results.append(row)

        return pd.DataFrame(results).sort_values(by=list(results[0].keys())[1],
                                                 ascending=False) if results else pd.DataFrame()

    def _plot_roc_curves(self, df, true_label_col, score_cols, filename, title_prefix):
        plt.figure(figsize=(8, 6))
        for score in score_cols:
            valid_data = df[[true_label_col, score]].dropna()
            if len(np.unique(valid_data[true_label_col])) < 2: continue
            fpr, tpr, _ = roc_curve(valid_data[true_label_col], valid_data[score])
            auc = roc_auc_score(valid_data[true_label_col], valid_data[score])
            plt.plot(fpr, tpr, lw=2, label=f'{score} (AUC = {auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'[{title_prefix}] ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=300)
        plt.close()

    def _plot_pr_curves(self, df, true_label_col, score_cols, filename, title_prefix):
        plt.figure(figsize=(8, 6))
        for score in score_cols:
            valid_data = df[[true_label_col, score]].dropna()
            if len(np.unique(valid_data[true_label_col])) < 2: continue
            precision, recall, _ = precision_recall_curve(valid_data[true_label_col], valid_data[score])
            auprc = average_precision_score(valid_data[true_label_col], valid_data[score])
            plt.plot(recall, precision, lw=2, label=f'{score} (AUPRC = {auprc:.2f})')

        baseline = df[true_label_col].mean()
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title(f'[{title_prefix}] Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=300)
        plt.close()

    # ----------------------------------------------------------------------
    # Longitudinal Engine
    # ----------------------------------------------------------------------
    def plot_longitudinal(self, df, true_label_col, score_cols, plot_type, settings):
        """Generates dynamic temporal performance plots."""
        bin_hours = settings.get('bin_hours', 24)
        max_hours = settings.get('max_hours', 168)
        metric_name = "AUROC" if plot_type == 'auroc_over_time' else "AUPRC"

        print(f"  📈 Generating {metric_name} Over Time (up to {max_hours}h)...")
        temporal_results = {score: {} for score in score_cols}
        bins = range(0, max_hours + bin_hours, bin_hours)

        for i in range(len(bins) - 1):
            start_h, end_h = bins[i], bins[i + 1]
            df_bin = df[(df['hours_from_admin'] >= start_h) & (df['hours_from_admin'] < end_h)]
            if df_bin.empty: continue

            df_bin = df_bin.groupby('patient_id')[score_cols + [true_label_col]].max().reset_index()
            y_true = df_bin[true_label_col]

            if len(np.unique(y_true)) > 1:
                for score in score_cols:
                    valid = df_bin[[true_label_col, score]].dropna()
                    if len(np.unique(valid[true_label_col])) > 1:
                        if metric_name == "AUROC":
                            val = roc_auc_score(valid[true_label_col], valid[score])
                        else:
                            val = average_precision_score(valid[true_label_col], valid[score])
                        temporal_results[score][end_h] = val

        # Plotting
        plt.figure(figsize=(12, 7))
        lines_plotted = 0
        for score in score_cols:
            x = list(temporal_results[score].keys())
            y = list(temporal_results[score].values())
            if len(x) > 0:
                plt.plot(x, y, marker='o', linewidth=2, label=score)
                lines_plotted += 1

        if lines_plotted == 0:
            print(f"  ❌ FAILED TO PLOT {metric_name}: Insufficient data points across time bins.")
            plt.close()
            return

        plt.title(f'{metric_name} Over Time ({bin_hours}h Bins)', fontsize=16, fontweight='bold')
        plt.xlabel('Hours Since Admission', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        baseline = 0.5 if metric_name == "AUROC" else df[true_label_col].mean()
        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7)
        plt.ylim(0.0 if metric_name == "AUPRC" else 0.4, 1.0)
        plt.xlim(0, max_hours)
        plt.xticks(bins[1:])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'temporal_{metric_name.lower()}.png', dpi=300)
        plt.close()