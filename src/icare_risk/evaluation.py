"""
Evaluation and Metrics Module (`icare_risk.evaluation`)
======================================================
Provides robust clinical and statistical classification metrics (AUROC, AUPRC,
Sensitivity, Specificity, PPV, NPV, F1, Youden's J, Confusion Matrix counts),
bootstrap confidence intervals, temporal performance tracking, and dynamic subgroup filtering.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    threshold: float = None,
                    target_sensitivity: float = None) -> dict:
    """
    Compute a comprehensive suite of clinical and statistical classification metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted probabilities.
    threshold : float, optional
        Fixed probability threshold. If None, uses Youden's J optimal threshold.
    target_sensitivity : float, optional
        If specified (e.g., 0.80), automatically finds the threshold that achieves
        at least this sensitivity, allowing standardized comparison across models.
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid_mask], y_pred[valid_mask]

    empty_res = {
        'auroc': np.nan, 'auprc': np.nan, 'brier_score': np.nan,
        'threshold': np.nan, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'sensitivity': np.nan, 'specificity': np.nan, 'ppv': np.nan,
        'npv': np.nan, 'accuracy': np.nan, 'balanced_accuracy': np.nan,
        'f1_score': np.nan, 'youden_j': np.nan
    }
    if len(np.unique(yt)) < 2:
        return empty_res

    # 1. Continuous probabilistic metrics
    results = {
        'auroc': roc_auc_score(yt, yp),
        'auprc': average_precision_score(yt, yp),
        'brier_score': brier_score_loss(yt, yp)
    }

    # 2. Threshold determination logic
    fpr, tpr, thresholds = roc_curve(yt, yp)

    if target_sensitivity is not None:
        # Find the lowest threshold that achieves at least target_sensitivity (e.g., 80%)
        valid_indices = np.where(tpr >= target_sensitivity)[0]
        if len(valid_indices) > 0:
            # Pick the index closest to target sensitivity (or highest specificity among those)
            best_idx = valid_indices[0]
            threshold = thresholds[best_idx]
        else:
            # Fallback to max sensitivity available if target is unattainable
            best_idx = np.argmax(tpr)
            threshold = thresholds[best_idx]
    elif threshold is None:
        # Default: Use Youden's J statistic (Sensitivity + Specificity - 1)
        best_idx = np.argmax(tpr - fpr)
        threshold = thresholds[best_idx]

    results['threshold'] = float(threshold)
    yp_binary = (yp >= threshold).astype(int)

    # 3. Confusion Matrix components
    cm = confusion_matrix(yt, yp_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    results.update({'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)})

    # 4. Derived clinical & statistical metrics
    sens = recall_score(yt, yp_binary, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results['sensitivity'] = sens
    results['specificity'] = spec
    results['ppv'] = precision_score(yt, yp_binary, zero_division=0)
    results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    results['accuracy'] = accuracy_score(yt, yp_binary)
    results['balanced_accuracy'] = balanced_accuracy_score(yt, yp_binary)
    results['f1_score'] = f1_score(yt, yp_binary, zero_division=0)
    results['youden_j'] = sens + spec - 1.0

    return results


def bootstrap_confidence_intervals(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   n_bootstraps: int = 1000,
                                   ci: float = 95,
                                   target_sensitivity: float = None) -> dict:
    """
    Compute bootstrap confidence intervals for all classification and clinical metrics.

    Performs non-parametric bootstrapping by resampling with replacement to estimate
    robust confidence intervals for AUROC, AUPRC, Brier score, sensitivity, specificity,
    PPV, NPV, and other computed metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities or risk scores.
    n_bootstraps : int, default=1000
        Number of bootstrap iterations to execute.
    ci : float, default=95
        Confidence interval percentage level (e.g., 95 for 95% CI).
    target_sensitivity : float, optional
        If specified (e.g., 0.80), fixes the sensitivity operating threshold during
        each bootstrap resampling iteration.

    Returns:
    --------
    dict
        A dictionary mapping each metric name with its lower and upper confidence bounds
        (e.g., `{'auroc_ci_lower': 0.78, 'auroc_ci_upper': 0.85}`).
    """
    np.random.seed(42)
    sample_res = compute_metrics(y_true, y_pred, target_sensitivity=target_sensitivity)
    metric_keys = [k for k in sample_res.keys() if k != 'threshold']

    boot_results = {m: [] for m in metric_keys}
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        res = compute_metrics(y_true[indices], y_pred[indices], target_sensitivity=target_sensitivity)
        for m in metric_keys:
            if m in res and not np.isnan(res[m]):
                boot_results[m].append(res[m])

    lower_pct, upper_pct = (100 - ci) / 2.0, 100 - ((100 - ci) / 2.0)
    ci_results = {}
    for m in metric_keys:
        vals = boot_results[m]
        ci_results[f"{m}_ci_lower"] = np.percentile(vals, lower_pct) if vals else np.nan
        ci_results[f"{m}_ci_upper"] = np.percentile(vals, upper_pct) if vals else np.nan

    return ci_results


def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """
    Dynamically filter a pandas DataFrame based on declarative YAML rule definitions.

    Supports various comparative and set-membership operators for cohort stratification
    and subgroup analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset to filter.
    filters : list of dict
        A list of filter rules parsed from YAML. Each dictionary must contain:
        - `column` (str): The column name to filter on.
        - `operator` (str): The logical operator (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`).
        - `value` (scalar or list): The value or collection of values to compare against.

    Returns:
    --------
    pd.DataFrame
        A filtered subset of the original DataFrame satisfying all specified conditions.

    Raises:
    -------
    KeyError
        If a specified filter column does not exist in the DataFrame.
    ValueError
        If an unsupported logical operator is provided.
    """
    subset = df.copy()
    op_map = {
        '==': lambda s, c, v: s[s[c] == v],
        '!=': lambda s, c, v: s[s[c] != v],
        '>': lambda s, c, v: s[s[c] > v],
        '>=': lambda s, c, v: s[s[c] >= v],
        '<': lambda s, c, v: s[s[c] < v],
        '<=': lambda s, c, v: s[s[c] <= v],
        'in': lambda s, c, v: s[s[c].isin(v)]
    }

    for f in filters:
        col, op, val = f['column'], f['operator'], f['value']
        if col not in subset.columns:
            raise KeyError(f"Filter column '{col}' not found in dataset.")
        if op not in op_map:
            raise ValueError(f"Unsupported operator: {op}")
        subset = op_map[op](subset, col, val)

    return subset


def evaluate_over_time(df: pd.DataFrame,
                       date_col: str,
                       target_col: str,
                       pred_col: str,
                       freq: str = '3M') -> pd.DataFrame:
    """
    Evaluate model performance across temporal bins to track stability and degradation over time.

    Groups data into custom temporal periods (e.g., monthly, quarterly, or yearly) and
    computes the full evaluation metric suite within each time window.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset containing admissions, dates, and predictions.
    date_col : str
        The name of the datetime column representing admission or event date.
    target_col : str
        The name of the ground truth target column.
    pred_col : str
        The name of the predicted probability column.
    freq : str, default='3M'
        Pandas time offset frequency string for grouping (e.g., '1M' for monthly,
        '3M' for quarterly / 3 months, '6M' for semi-annually, '1Y' for yearly).

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame where each row corresponds to a temporal period, reporting
        the period start date, sample size (`n_samples`), number of positive events (`n_events`),
        and all computed classification metrics.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    temporal_records = []
    for period, group in df.groupby(pd.Grouper(key=date_col, freq=freq)):
        if len(group) == 0:
            continue
        metrics = compute_metrics(group[target_col].values, group[pred_col].values)
        temporal_records.append({
            'period_start': period.strftime('%Y-%m-%d'),
            'n_samples': len(group),
            'n_events': int(group[target_col].sum()),
            **metrics
        })

    return pd.DataFrame(temporal_records)