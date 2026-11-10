import pandas as pd
from typing import List, Optional

DEFAULT_KEYS = ["SUBJECT", "SPELL_IDENTIFIER"]

def keep_first_spell(df: pd.DataFrame, 
                     name: str = "episodes",
                     keys: Optional[List[str]] = DEFAULT_KEYS
) -> pd.DataFrame:
    """Keep the first source row for each SUBJECT/SPELL_IDENTIFIER pair."""
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing key columns: {missing}")
    result = df.copy()
    before = len(result)
    result = result.drop_duplicates(keys, keep="first").reset_index(drop=True)
    print(f"{name}: {before:,} rows -> {len(result):,} unique spells")
    return result