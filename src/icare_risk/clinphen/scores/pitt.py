import pandas as pd
import numpy as np

from typing import Dict, Optional, List

def calculate_pitt_score(df: pd.DataFrame,
                       age_col: str = "AGE_AT_ADMISSION",
                       weights: Optional[Dict[str, int]] = None,
                       verbose: int = 0) -> pd.Series:
    """
    Computes the Pitt Bacteremia Score (0 to 14 points).

    The Pitt Bacteremia Score is a validated clinical instrument used to measure acute
    illness severity in patients experiencing bloodstream infections. By evaluating
    parameters such as mental status, fever status, hypotension, mechanical ventilation,
    and recent cardiac arrest, it assigns a discrete point score (typically ranging from
    0 to 14) that helps clinicians estimate short-term mortality risk and guide urgent
    therapeutic decisions.

    !!! warning "Pre-computed Inputs Required"
        This function expects **pre-calculated points** and binary flags, NOT raw clinical values.
        For example, `temp_col` must contain the discrete Pitt points (0, 1, or 2), not the raw
        temperature in Celsius. Passing raw vitals will result in massive calculation errors.

    ??? note "Clinical Logic & Point Allocation (Click to expand)"
        Because 'temp' and 'mental' phenotypes already return the exact points (0, 1, 2, 4),
        we multiply them by 1. The boolean flags get multiplied by their specific Pitt weights.

        * **Fever Status:** +1 or +2 points (Pre-computed input required)
        * **Mental Status:** +1, +2, or +4 points (Pre-computed input required)
        * **Hypotension:** +2 points (Requires binary 1/0 flag)
        * **Mechanical Ventilation:** +2 points (Requires binary 1/0 flag)
        * **Cardiac Arrest:** +4 points (Requires binary 1/0 flag)

        Missing columns are safely handled and default to 0 points.

        **References:**  Paterson, D. L. et al International prospective study of Klebsiella pneumoniae
        bacteremia: implications of extended-spectrum beta-lactamase production in nosocomial infections.
        Annals of Internal Medicine, 140(1), 26–32. (2004)

    """
    pass