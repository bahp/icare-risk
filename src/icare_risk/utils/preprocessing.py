import pandas as pd
from typing import List, Optional

DEFAULT_KEYS = ["SUBJECT", "SPELL_IDENTIFIER"]

def keep_unique_rows(df: pd.DataFrame, 
                     keys: list[str], 
                     name: str) -> pd.DataFrame:
    """Keep the first source row for the given granularity keys."""
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing key columns: {missing}")
    result = df.copy()
    before = len(result)
    result = result.drop_duplicates(keys, keep="first").reset_index(drop=True)
    print(f"{name} ({'-'.join(keys)}): {before:,} rows -> {len(result):,} unique records")
    return result

def keep_first_spell(df: pd.DataFrame, 
                     name: str = "episodes",
                     keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Keep the first source row for each SUBJECT/SPELL_IDENTIFIER pair."""
    # Use DEFAULT_KEYS if keys are not provided
    keys = keys if keys is not None else DEFAULT_KEYS
    
    # Delegate directly to keep_unique_rows
    return keep_unique_rows(df, keys=keys, name=name)