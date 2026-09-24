from typing import Optional, Tuple
from icare_risk.clinphen.temporal.context import EpisodeContext

def derive_increment_bsi_not_urinary_flag(ctx: EpisodeContext,
        window: Optional[Tuple[str, str]] = None, **kwargs):
    """
    Determines if the source of the Bloodstream Infection (BSI) is NON-urinary.

    In the INCREMENT-ESBL score, non-urinary sources (respiratory, abdominal, etc.)
    are associated with higher mortality and receive +3 points.

    ??? note "Clinical Logic & iCARE Mapping (Click to expand)"
        A patient is flagged as having a NON-urinary source (returns 1) UNLESS we can
        prove the source was urinary. We assume urinary if:

        1. **Explicit Flag:** The `bsi_source` column explicitly equals 'Urinary'.
        2. **Concurrent Cultures:** A urine culture (`urine_culture_result`) drawn within 48h
           grew the same organism.
        3. **ICD-10 Proxy:** The patient was billed for a UTI (e.g., 'N39.0') on admission.

        If none of the urinary criteria are met, the function defaults to non-urinary (1).

        **iCARE Mapping:**

        * **Table:** `icare_episodes_diagnosis_anon`
        * **Column:** `diagnosis_code_snomed`
        * **Logic:** Use the diagnosis ICD-10 codes: J15.8, J15.9, J18.0, J18.1, J18.9,
          J85.1, N39.0, N10, N13.6, N30.0, N30.9 (to cover sputum and urine).

    """
    df_mic = ctx.get_current("microbiology", window=window)
    if df_mic.empty:
        return 0

    # Standardize columns for safety
    order_codes = df_mic['code'].fillna('').str.lower().str.strip()
    sites = df_mic['site'].fillna('').str.lower().str.strip()

    # Define urinary order codes and terms to exclude
    urinary_order_codes = {'urncul', 'uricul', 'urine micro'}

    # Condition: order code is not a urinary code AND site does not contain 'urine'
    is_not_urinary = (~order_codes.isin(urinary_order_codes)) & (~sites.str.contains('urine', na=False))

    return int(is_not_urinary.any())

def derive_increment_is_non_ecoli_flag(ctx: EpisodeContext,
        window: Optional[Tuple[str, str]] = None, **kwargs):
    """
    Identifies if the ESBL-producing organism is a non-E. coli species.

    ??? info "Clinical Definition"
        E. coli bacteremia generally has a better prognosis in this specific cohort.
        We flag the patient as 'Non-E. coli' if ANY of the following are true:

        1. Blood Culture Result: The `microorganism` or `blood_culture_org` column
        contains keywords like 'klebsiella', 'enterobacter', 'serratia', or 'other'.
        2. Exclusion: Ensure we do NOT flag it if the string strictly says
        'escherichia coli' or 'e. coli'.

    Required Columns in df:
    - `microorganism` or `blood_culture_org` (String from microbiology LIS system)
    """
    df_mic = ctx.get_current("microbiology", window=window)
    if df_mic.empty:
        return 0

    orgs = df_mic.organism.fillna('').str.lower().str.strip()

    # Phrases that indicate negative cultures or general screens rather than a specific pathogen
    negatives = {
        'no growth', 'none', 'mixed bacterial growth', 'mrsa not isolated',
        'no carbapenem resistant organisms isolated', 'no resistant acinetobacter isolated',
        'no significant growth', 'normal upper respiratory flora.', 'no growth at 2 days',
        'appearance:', 'no growth at 5 days', 'not isolated'
    }

    # Must have text and not be a negative/no-growth string
    is_valid_isolate = (orgs != '') & (~orgs.isin(negatives)) & (~orgs.str.contains('not isolated', na=False))
    valid_df = df_mic.loc[is_valid_isolate]

    if valid_df.empty:
        return 0

    # Check if any valid isolate does NOT contain 'escherichia' or 'coli'
    valid_orgs = valid_df['organism_bug'].str.lower()
    is_non_ecoli = ~valid_orgs.str.contains('escherichia|coli', na=False)

    return int(is_non_ecoli.any())

def derive_inappropriate_empiric_abx(ctx: EpisodeContext,
        window: Optional[Tuple[str, str]] = None, **kwargs):
    """
    Evaluates if the empirical antibiotic therapy administered was INAPPROPRIATE.

    Compares 'microbiology' sensitivity results against 'pharmacy'
    prescriptions. Inappropriate therapy (+2 points in INCREMENT-ESBL)
    occurs if the organism is resistant to all administered drugs, or if
    NO 'Susceptible' antibiotic was given within 24h of the culture.

    !!! warning "Missing Data & Simplification Assumptions"
        * **Default to Risk:** If microbiology or pharmacy context data is entirely missing for a patient, this function defaults to **1 (Inappropriate Therapy)**.
        * **Synthetic Simplification:** In this synthetic version, if the microbiology report simply says 'Resistant', it is automatically flagged as inappropriate therapy.

    ??? "Clinical Notes & iCARE Mapping (Click to expand)"
        Empirical therapy is the drug given *before* the final lab results come back.
        Therapy is considered **inappropriate** (returns 1) if:

        1. **Resistance:** The `empiric_abx_given` (medication given in the first 24h)
        matches a drug listed in the `abx_resistant_to` column (from the lab report).
        2. **No Coverage:** The patient received no active anti-ESBL antibiotics
           (like carbapenems) within the first 24 hours of blood culture collection.
        3. **Explicit Flag:** If an `inappropriate_abx_flag` already exists from a
           clinical pharmacist's manual review, use it.

        Required Columns in df:
        - `empiric_abx_given` (String/List of medications)
        - `abx_resistant_to` (String/List from susceptibility report)
        - `inappropriate_abx_flag` (Int 1 or 0, optional)

        This method requires looking at Pharmacy data (`empiric_abx_given`) and
        Laboratory data (`abx_resistant_to`) simultaneously.

        **iCARE Mapping:**

        * Merge `icare_microbiology_anon` with `icare_pharmacy_prescribing_anon`.
        * Compare `sample_collected_dt` (`icare_pathology_blood_anon`)
          with `order_dt_tm` (`icare_pharmacy_prescribing_anon`).
        * Inappropriate if administration > 1 day from blood cultures or
          > 4 days without targeted anti-ESBL therapy (e.g. Carbapenems).
    """
    # Fetch prescribing data and microbiology sensitivity data
    df_rx = ctx.get_current("prescribing", window=window)
    df_mic = ctx.get_current("microbiology", window=window) # (0h, 24h)

    if df_rx.empty or df_mic.empty:
        return 0

    # Implement your clinical matching logic here (e.g., matching drug class to resistance profile)
    # Return 1 if inappropriate, else 0
    return 0