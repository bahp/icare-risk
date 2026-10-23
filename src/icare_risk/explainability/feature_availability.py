from __future__ import annotations

from collections.abc import Mapping, Sequence
import pandas as pd


def audit_feature_availability(
    df: pd.DataFrame,
    feature_sets: Mapping[str, Mapping],
    *,
    prediction_time: str = "admission",
    known_unavailable: Sequence[str] = (),
    outcome_columns: Sequence[str] = (),
    post_prediction_columns: Sequence[str] = (),
    suspicious_terms: Sequence[str] = ("culture", "outcome", "esbl_status", "label", "discharge", "death", "update"),
) -> pd.DataFrame:
    """Create an auditable feature-level availability/leakage table."""
    rows = []
    for set_name, spec in feature_sets.items():
        if spec.get("include_all"):
            columns = list(df.columns)
        else:
            columns = list(spec.get("columns", []))
        columns = [c for c in columns if c in df.columns]
        exclusions = set(spec.get("exclude", []))
        for column in columns:
            lower = column.lower()
            explicit_outcome = column in set(outcome_columns)
            explicit_post = column in set(post_prediction_columns) or column in set(known_unavailable)
            suspicious = any(term.lower() in lower for term in suspicious_terms)
            if explicit_outcome:
                decision, reason = "exclude", "outcome or outcome-derived variable"
            elif explicit_post:
                decision, reason = "exclude", "unavailable after prediction time"
            elif column in exclusions:
                decision, reason = "exclude", "configured exclusion"
            elif suspicious:
                decision, reason = "review", "name suggests possible leakage or timing issue"
            else:
                decision, reason = "review", "requires clinical confirmation"
            rows.append({"feature_set": set_name, "feature": column, "present": True, "decision": decision, "reason": reason, "prediction_time": prediction_time})
    return pd.DataFrame(rows).drop_duplicates(["feature_set", "feature"])


def audit_model_features(features: Sequence[str], *, forbidden: Sequence[str] = ()) -> pd.DataFrame:
    forbidden_set = set(forbidden)
    return pd.DataFrame({"feature": list(features), "forbidden": [f in forbidden_set for f in features], "decision": ["exclude" if f in forbidden_set else "review" for f in features]})
