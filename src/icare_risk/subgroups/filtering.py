from __future__ import annotations

from collections.abc import Iterable, Mapping
import pandas as pd

from .definitions import SubgroupDefinition


def _condition_mask(df: pd.DataFrame, condition: Mapping, subgroup_name: str) -> pd.Series:
    column = condition["column"]
    if column not in df.columns:
        raise KeyError(f"Subgroup column '{column}' not found in dataset.")

    series = df[column]
    op = condition["operator"]
    value = condition.get("value")

    if op == "==":
        mask = series == value
    elif op == "!=":
        mask = series != value
    elif op == ">":
        mask = series > value
    elif op == ">=":
        mask = series >= value
    elif op == "<":
        mask = series < value
    elif op == "<=":
        mask = series <= value
    elif op == "in":
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise TypeError(f"Value for 'in' subgroup '{subgroup_name}' must be a collection.")
        mask = series.isin(value)
    elif op == "not_in":
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise TypeError(f"Value for 'not_in' subgroup '{subgroup_name}' must be a collection.")
        mask = ~series.isin(value)
    elif op == "notna":
        mask = series.notna()
    elif op == "isna":
        mask = series.isna()
    else:
        raise ValueError(f"Unsupported subgroup operator: {op}")

    return mask.fillna(False)


def subgroup_mask(df: pd.DataFrame, definition: SubgroupDefinition) -> pd.Series:
    """Return a boolean mask for simple or compound subgroup definitions."""
    if definition.is_compound:
        masks = [_condition_mask(df, condition, definition.name)
                 for condition in definition.conditions]
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined & mask if definition.logic == "all" else combined | mask
        return combined.fillna(False)

    return _condition_mask(df, {
        "column": definition.column,
        "operator": definition.operator,
        "value": definition.value,
    }, definition.name)


def apply_subgroup(df: pd.DataFrame, definition: SubgroupDefinition) -> pd.DataFrame:
    """Return rows belonging to one subgroup."""
    return df.loc[subgroup_mask(df, definition)].copy()
