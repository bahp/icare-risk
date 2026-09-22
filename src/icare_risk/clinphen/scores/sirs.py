import pandas as pd
import numpy as np

from typing import Dict, Optional, List

def calculate_sirs_score(df: pd.DataFrame,
                       age_col: str = "AGE_AT_ADMISSION",
                       weights: Optional[Dict[str, int]] = None,
                       verbose: int = 0) -> pd.Series:
    """
        Computes the number of SIRS criteria met (0 to 4).

        The Systemic Inflammatory Response Syndrome (SIRS) score is a clinical screening
        tool used to identify generalized systemic inflammation and severe stress responses
        to conditions like infection, trauma, or burns. By evaluating four physiological
        parameters—heart rate, respiratory rate, core body temperature, and white blood
        cell count—it measures whether a patient meets criteria for widespread inflammatory
        activation, which historically served as a foundational framework for defining sepsis.

        !!! warning "Binary Inputs Required"
            This function strictly requires **binary flags (1 or 0)** for each of the four
            SIRS criteria. It does NOT evaluate raw vital signs (like heart rate > 90).
            Raw data must be translated into binary flags upstream in `phenotypes.py`.

        ??? note "Calculation Logic (Click to expand)"
            Simply adds up the pre-computed 1/0 flags for Tachycardia, Tachypnea,
            Abnormal Temperature, and Abnormal WBC. Missing data is treated as 0
            (condition not met).

            **References:** Bone, R. C. (1992). Definitions for sepsis and organ failure
            and guidelines for the use of innovative therapies in sepsis. Chest, 101(6),
            1644–1655.
    """
    pass