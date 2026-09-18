# Libraries
import pandas as pd

def has_diabetes(
        df: pd.DataFrame,
        history_flag_col: str = "hx_diabetes",
        rx_flag_col: str = "rx_antidiabetics",
        glucose_max_col: str = "glucose_max"
) -> pd.Series:
    """
    Vectorized Charlson Diabetes check using multimodal proxies[cite: 2].
    Evaluates explicit history, medication proxies, and lab value proxies[cite: 2].
    """
    is_diabetic = pd.Series(False, index=df.index)

    # 1. Explicit History Check
    if history_flag_col in df.columns:
        is_diabetic = is_diabetic | (df[history_flag_col] == 1)

    # 2. Medication Proxy Check (assuming user has a boolean flag for these meds)
    if rx_flag_col in df.columns:
        is_diabetic = is_diabetic | (df[rx_flag_col] == 1)

    # 3. Lab Values Proxy (Rolling max glucose > 200 mg/dL)[cite: 2]
    if glucose_max_col in df.columns:
        is_diabetic = is_diabetic | (df[glucose_max_col] > 200.0)

    return is_diabetic.astype(int)