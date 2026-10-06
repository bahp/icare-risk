from typing import Dict, Optional
import pandas as pd


def calculate_weighted_points(
    df: pd.DataFrame,
    weights: Dict[str, int],
) -> pd.Series:
    """Calculate a row-wise weighted score from dataframe columns."""
    cols = [col for col in weights if col in df.columns]

    if not cols:
        return pd.Series(0, index=df.index, dtype=float)

    values = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    weight_series = pd.Series({col: weights[col] for col in cols})

    return values.dot(weight_series)


def log_score_audit(
    df: pd.DataFrame,
    weights: Dict[str, int],
    score: pd.Series,
    title: str = "Score Breakdown",
    verbose: int = 0,
) -> None:
    """Print a compact audit trace for the first row."""
    if verbose <= 0:
        return

    print(f"  [{title}]")

    if len(df) > 1:
        print(f"    ⚠️ DataFrame contains {len(df)} rows; showing row 0 only.")

    for col, weight in weights.items():
        if col not in df.columns:
            if verbose >= 2:
                print(f"    [?] {col:<32} (Missing Col) +0")
            continue

        value = pd.to_numeric(df[col].iloc[0], errors="coerce")
        value = 0 if pd.isna(value) else value
        points = value * weight

        if verbose >= 2 or points != 0:
            print(f"    [{'+' if points else ' '}] {col:<32} "
                  f"(Value: {str(value):<5}) +{points:g}")

    total = score.iloc[0] if len(score) else 0
    print(f"    {'-' * 55}")
    print(f"    [=] TOTAL COMPUTED SCORE:         {total:g}\n")
