from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def transformed_data(fitted_pipeline, X: pd.DataFrame):
    """Transform raw features using the fitted preprocessing step."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return preprocessor.transform(X)


def transformed_feature_names(fitted_pipeline) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def calculate_shap_values(fitted_pipeline, X: pd.DataFrame):
    """Calculate SHAP values for a fitted sklearn pipeline.

    For tree classifiers this uses TreeExplainer; for linear classifiers it uses
    LinearExplainer. The returned values explain the classifier output on the
    transformed feature scale.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to run SHAP analysis: pip install shap") from exc

    transformed = transformed_data(fitted_pipeline, X)
    classifier = fitted_pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        # SHAP's TreeExplainer can fail with some XGBoost versions when the
        # model stores base_score as a one-element vector. Use XGBoost's
        # native pred_contribs API as a compatible fallback.
        try:
            explainer = shap.TreeExplainer(classifier)
            explanation = explainer(transformed)
            values = explanation.values
        except (TypeError, ValueError, AttributeError) as exc:
            if not hasattr(classifier, "get_booster"):
                raise
            try:
                import xgboost as xgb
            except ImportError as xgb_exc:
                raise ImportError(
                    "XGBoost is required for the native SHAP fallback."
                ) from xgb_exc

            booster = classifier.get_booster()
            contributions = booster.predict(
                xgb.DMatrix(transformed),
                pred_contribs=True,
            )
            # The final column is the expected-value/base contribution.
            values = np.asarray(contributions)[..., :-1]

    elif hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, transformed)
        values = explainer(transformed).values
    else:
        explainer = shap.Explainer(classifier, transformed)
        values = explainer(transformed).values

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values, transformed_feature_names(fitted_pipeline)


def shap_importance_table(shap_values, feature_names: Sequence[str], *, model_name: str | None = None) -> pd.DataFrame:
    """Create a mean absolute SHAP ranking table."""
    values = np.asarray(shap_values)
    table = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table["importance_rank"] = range(1, len(table) + 1)
    if model_name is not None:
        table.insert(0, "model", model_name)
    return table


def shap_values_table(shap_values, feature_names: Sequence[str], index=None) -> pd.DataFrame:
    """Return row-level SHAP values in a tidy dataframe."""
    values = np.asarray(shap_values)
    return pd.DataFrame(values, columns=list(feature_names), index=index)
