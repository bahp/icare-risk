"""Reusable model explainability utilities."""

from .feature_importance import (
    model_coefficients,
    native_tree_importance,
    permutation_importance_table,
    transformed_feature_names,
)
from .shap_analysis import (
    calculate_shap_values,
    shap_importance_table,
    shap_values_table,
)
from .sensitivity import one_at_a_time_sensitivity, prediction_sensitivity
from .counterfactuals import evaluate_counterfactuals

__all__ = [
    "model_coefficients",
    "native_tree_importance",
    "permutation_importance_table",
    "transformed_feature_names",
    "calculate_shap_values",
    "shap_importance_table",
    "shap_values_table",
    "prediction_sensitivity",
    "one_at_a_time_sensitivity",
    "evaluate_counterfactuals",
]
