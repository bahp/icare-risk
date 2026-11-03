from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .clinical_scores import add_configured_scores, score_summary, validate_score_columns


def prepare_scores(
    df: pd.DataFrame,
    score_definitions: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare configured clinical scores and return data plus validation summary."""
    prepared = add_configured_scores(df, score_definitions)
    score_columns = list(score_definitions)
    validation = validate_score_columns(prepared, score_columns)
    return prepared, validation


def configured_score_summary(
    df: pd.DataFrame,
    score_definitions: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Return distribution summaries for the configured score columns."""
    return score_summary(df, score_definitions.keys())
