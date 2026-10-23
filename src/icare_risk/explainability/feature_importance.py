from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.inspection import permutation_importance


def transformed_feature_names(fitted_pipeline) -> list[str]:
    """Return feature names produced by a fitted sklearn preprocessing pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return [f"feature_{i}" for i in range(preprocessor.transform(pd.DataFrame([[0] * 0])).shape[1])]


def model_coefficients(fitted_pipeline, *, model_name: str | None = None) -> pd.DataFrame:
    """Extract logistic-regression coefficients from a fitted pipeline."""
    classifier = fitted_pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise TypeError("The fitted classifier does not expose logistic-regression coefficients.")
    names = transformed_feature_names(fitted_pipeline)
    coefficients = classifier.coef_[0]
    result = pd.DataFrame({"feature": names, "coefficient": coefficients})
    result["odds_ratio"] = result["coefficient"].map(lambda value: float(__import__("numpy").exp(value)))
    result["absolute_coefficient"] = result["coefficient"].abs()
    if model_name is not None:
        result.insert(0, "model", model_name)
    return result.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    fitted_pipeline,
    X: pd.DataFrame,
    y,
    *,
    model_name: str | None = None,
    scoring: str = "roc_auc",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate held-out permutation importance on the original feature columns."""
    result = permutation_importance(
        fitted_pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table["importance_rank"] = range(1, len(table) + 1)
    if model_name is not None:
        table.insert(0, "model", model_name)
    return table


def native_tree_importance(fitted_pipeline, *, model_name: str | None = None) -> pd.DataFrame:
    """Extract native tree importances when the classifier provides them."""
    classifier = fitted_pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("The fitted classifier does not expose native feature importances.")
    names = transformed_feature_names(fitted_pipeline)
    table = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["importance_rank"] = range(1, len(table) + 1)
    if model_name is not None:
        table.insert(0, "model", model_name)
    return table
