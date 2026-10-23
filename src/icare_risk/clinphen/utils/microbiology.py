import pandas as pd

def compute_esbl_labels(micro_df,
                        path_3gcr='3GCR.csv',
                        path_3gcs='3GCS.csv',
                        group_cols=['SUBJECT', 'LAB_TEST_ID']):
    """
    Computes and assigns Extended-Spectrum Beta-Lactamase (ESBL) status labels
    to microbiology records and test episodes.

    This function matches individual antibiotic susceptibility results against
    predefined reference resistance (3GCR) and susceptibility (3GCS) files,
    incorporates explicit ESBL lab markers, and aggregates findings at the
    specified group level (e.g., per specimen test ID or patient episode).

    Parameters:
    -----------
    micro_df : pandas.DataFrame
        The raw microbiology dataframe containing columns 'SENSITIVITY'
        and the grouping columns.
    path_3gcr : str, default '3GCR.csv'
        File path to the 3rd-generation cephalosporin resistance lookup table.
    path_3gcs : str, default '3GCS.csv'
        File path to the 3rd-generation cephalosporin susceptibility lookup table.
    group_cols : list of str, default ['SUBJECT', 'LAB_TEST_ID']
        Columns used to group row-level antimicrobial results into distinct
        test events or clinical episodes.

    Returns:
    --------
    pandas.DataFrame
        A dataframe containing the grouping columns and the computed
        'esbl_status' column with values 'ESBL', 'non-ESBL', or 'unknown'.

    Example:
    --------
        >>> esbl_results = compute_esbl_labels(microbiology_df)
        >>> final_df = microbiology_df.merge(esbl_results, on=['SUBJECT', 'LAB_TEST_ID'], how='left')
    """
    # 1. Load and parse 3GCR and 3GCS reference files into lookup sets
    def _parse_lookup(path):
        df = pd.read_csv(path)
        parts = df['SENSITIVITY'].str.split(':', n=1, expand=True)
        a = parts[0].str.strip().str.lower()
        r = parts[1].str.strip().str.lower() if parts.shape[1] > 1 else pd.Series([""] * len(df))
        return set(zip(a, r))

    r_set = _parse_lookup(path_3gcr)
    s_set = _parse_lookup(path_3gcs)

    # 2. Parse microbiology dataset SENSITIVITY column
    df = micro_df.copy()
    split_micro = df['SENSITIVITY'].str.split(':', n=1, expand=True)
    df['antibiotic'] = split_micro[0].str.strip().str.lower()
    df['result'] = split_micro[1].str.strip().str.lower() if split_micro.shape[1] > 1 else ""
    df['abx_result_pair'] = list(zip(df['antibiotic'], df['result']))

    # 3. Evaluate criteria per row
    marker_pos_terms = {
        "sensitive",
        "susceptible",
        "susceptible with optimised dosing, refer to antimicrobial policy"
    }

    df['is_3gcr'] = df['abx_result_pair'].isin(r_set)
    df['is_3gcs'] = df['abx_result_pair'].isin(s_set)
    df['is_marker_pos'] = (df['antibiotic'] == 'esbl markers (ss = present)') & (df['result'].isin(marker_pos_terms))
    df['is_marker_neg'] = (df['antibiotic'] == 'esbl markers (ss = present)') & (df['result'] == 'resistant')

    # 4. Aggregate by episode/test level
    agg_df = df.groupby(group_cols).agg(
        has_3gcr=('is_3gcr', 'any'),
        has_3gcs=('is_3gcs', 'any'),
        has_marker_pos=('is_marker_pos', 'any'),
        has_marker_neg=('is_marker_neg', 'any')
    ).reset_index()

    def determine_status(row):
        if row['has_3gcr'] or row['has_marker_pos']:
            return 'ESBL'
        elif row['has_3gcs'] or row['has_marker_neg']:
            return 'non-ESBL'
        return 'unknown'

    agg_df['esbl_status'] = agg_df.apply(determine_status, axis=1)

    return agg_df[group_cols + ['esbl_status']]


def assign_esbl_to_episodes(
        df_episodes,
        df_microbiology,
        path_3gcr='3GCR.csv',
        path_3gcs='3GCS.csv',
        pre_admission_days=14,
        post_discharge_days=7,
        spell_col='SPELL_IDENTIFIER'
):
    """
    Links test-level ESBL labels to hospital *spells/admissions* (one label per
    `spell_col`, e.g. SPELL_IDENTIFIER), then broadcasts that label back onto
    every row of `df_episodes` belonging to that spell.

    Why spell-level, not episode-row-level:
    Spells can contain several episode rows (ward moves, specialty changes).
    Windowing each row independently and then re-grouping on *all* columns of
    df_episodes (as before) is fragile and can give inconsistent ESBL labels
    for rows that belong to the same admission. Here we first collapse each
    spell to a single [min(ADMISSION_DATE), max(DISCHARGE_DATE)] window,
    attach microbiology tests to that window, decide one status per spell,
    then merge it back onto every episode row of that spell.

    Parameters
    ----------
    df_episodes : pandas.DataFrame
        Must contain 'SUBJECT', spell_col, 'ADMISSION_DATE', 'DISCHARGE_DATE'.
        May contain multiple rows per spell_col.
    df_microbiology : pandas.DataFrame
        Must contain 'SUBJECT', 'LAB_TEST_ID', 'LATEST_COLLECT_DT', 'SENSITIVITY'
        (or pre-computed 'esbl_status').
    path_3gcr, path_3gcs : str
        Lookup file paths, forwarded to compute_esbl_labels.
    pre_admission_days, post_discharge_days : int
        Window padding around the spell.
    spell_col : str, default 'SPELL_IDENTIFIER'
        Column identifying a single hospital admission/spell.

    Returns
    -------
    pandas.DataFrame
        df_episodes with an added 'esbl_status' column ('ESBL', 'non-ESBL',
        'unknown'), identical for every row sharing the same spell_col value.
    """
    ep_df = df_episodes.copy()
    micro_df = df_microbiology.copy()

    # 1. Normalize SUBJECT (and spell key) to avoid int/str merge mismatches
    ep_df['SUBJECT'] = ep_df['SUBJECT'].astype(str).str.strip()
    micro_df['SUBJECT'] = micro_df['SUBJECT'].astype(str).str.strip()
    ep_df[spell_col] = ep_df[spell_col].astype(str).str.strip()

    # 2. Compute test-level ESBL status if not already present
    if 'esbl_status' not in micro_df.columns:
        print("Computing test-level ESBL labels on the fly...")
        esbl_results = compute_esbl_labels(micro_df, path_3gcr=path_3gcr, path_3gcs=path_3gcs)
        micro_df = micro_df.merge(esbl_results, on=['SUBJECT', 'LAB_TEST_ID'], how='left')

    test_merged = micro_df[['SUBJECT', 'LAB_TEST_ID', 'LATEST_COLLECT_DT', 'esbl_status']].drop_duplicates()
    test_merged['LATEST_COLLECT_DT'] = pd.to_datetime(test_merged['LATEST_COLLECT_DT'])

    ep_df['ADMISSION_DATE'] = pd.to_datetime(ep_df['ADMISSION_DATE'])
    ep_df['DISCHARGE_DATE'] = pd.to_datetime(ep_df['DISCHARGE_DATE'])

    # 3. Collapse episode rows to ONE window per spell (admission)
    spell_windows = ep_df.groupby(['SUBJECT', spell_col], as_index=False).agg(
        SPELL_ADMISSION_DATE=('ADMISSION_DATE', 'min'),
        SPELL_DISCHARGE_DATE=('DISCHARGE_DATE', 'max')
    )

    # 4. Attach candidate tests to each spell (join on SUBJECT only, then filter by window)
    joined = spell_windows.merge(test_merged, on='SUBJECT', how='left')

    in_window = (
        (joined['LATEST_COLLECT_DT'] >= (joined['SPELL_ADMISSION_DATE'] - pd.Timedelta(days=pre_admission_days))) &
        (joined['LATEST_COLLECT_DT'] <= (joined['SPELL_DISCHARGE_DATE'] + pd.Timedelta(days=post_discharge_days)))
    )
    valid = joined[in_window].copy()

    # 5. Priority-based aggregation: ESBL > non-ESBL > unknown
    priority_map = {'ESBL': 2, 'non-ESBL': 1, 'unknown': 0}
    valid['priority'] = valid['esbl_status'].map(priority_map)

    spell_status = valid.groupby(['SUBJECT', spell_col])['priority'].max().reset_index()
    reverse_priority_map = {2: 'ESBL', 1: 'non-ESBL', 0: 'unknown'}
    spell_status['esbl_status'] = spell_status['priority'].map(reverse_priority_map)
    spell_status = spell_status[['SUBJECT', spell_col, 'esbl_status']]

    # 6. Attach status to every spell (spells with no matching test -> 'unknown')
    spell_windows = spell_windows.merge(spell_status, on=['SUBJECT', spell_col], how='left')
    spell_windows['esbl_status'] = spell_windows['esbl_status'].fillna('unknown')

    # 7. Broadcast spell-level label back onto every original episode row
    final_episodes = ep_df.merge(
        spell_windows[['SUBJECT', spell_col, 'esbl_status']],
        on=['SUBJECT', spell_col],
        how='left'
    )
    final_episodes['esbl_status'] = final_episodes['esbl_status'].fillna('unknown')

    return final_episodes


import pandas as pd


def compute_time_to_closest_culture(
        df_episodes,
        microbiology_df,
        pre_admission_days=14,
        post_discharge_days=7,
        round_decimals=1
):
    """
    Computes the time from episode admission to the closest microbiology culture collection.

    Parameters:
    -----------
    df_episodes : pandas.DataFrame
        Episodes table containing 'SUBJECT', 'ADMISSION_DATE', 'ADMISSION_TIME' (optional).
    microbiology_df : pandas.DataFrame
        Microbiology table containing 'SUBJECT', 'LAB_TEST_ID', 'LATEST_COLLECT_DT'.
    pre_admission_days : int, default 14
        Include cultures collected up to X days before admission.
    post_discharge_days : int, default 7
        Include cultures collected up to X days after discharge.

    Returns:
    --------
    pandas.DataFrame
        The episodes DataFrame with added columns:
        - 'closest_culture_date': The collection date of the closest culture.
        - 'time_to_culture_days': Floating-point days from admission to culture collection
          (negative values mean pre-admission cultures).
        - 'time_to_culture_hours': Total hours from admission to culture collection.
    """
    ep_df = df_episodes.copy()
    micro_df = microbiology_df.copy()

    # 1. Standardize patient IDs to prevent type mismatches (int vs str)
    ep_df['SUBJECT'] = ep_df['SUBJECT'].astype(str).str.strip()
    micro_df['SUBJECT'] = micro_df['SUBJECT'].astype(str).str.strip()

    # 2. Construct full admission datetime if time is available
    if 'ADMISSION_TIME' in ep_df.columns and ep_df['ADMISSION_TIME'].notna().any():
        ep_df['admission_datetime'] = pd.to_datetime(
            ep_df['ADMISSION_DATE'].astype(str) + ' ' + ep_df['ADMISSION_TIME'].astype(str),
            errors='coerce'
        )
    else:
        ep_df['admission_datetime'] = pd.to_datetime(ep_df['ADMISSION_DATE'], errors='coerce')

    ep_df['discharge_datetime'] = pd.to_datetime(ep_df['DISCHARGE_DATE'], errors='coerce')
    micro_df['collect_datetime'] = pd.to_datetime(micro_df['LATEST_COLLECT_DT'], errors='coerce')

    # Extract unique collection timestamps per subject/test to avoid duplicate row explosion
    test_dates = micro_df[['SUBJECT', 'LAB_TEST_ID', 'collect_datetime']].drop_duplicates().dropna(
        subset=['collect_datetime'])

    # 3. Merge episodes with microbiology records on SUBJECT
    joined = ep_df.merge(test_dates, on='SUBJECT', how='left', suffixes=('', '_test'))

    # 4. Filter by the allowable window (pre-admission buffer to post-discharge buffer)
    window_start = joined['admission_datetime'] - pd.Timedelta(days=pre_admission_days)
    window_end = joined['discharge_datetime'] + pd.Timedelta(days=post_discharge_days)

    in_window = (joined['collect_datetime'] >= window_start) & (joined['collect_datetime'] <= window_end)
    valid_tests = joined[in_window].copy()

    if valid_tests.empty:
        ep_df['closest_culture_date'] = pd.NaT
        ep_df['time_to_culture_days'] = None
        ep_df['time_to_culture_hours'] = None
        return ep_df

    # 5. Calculate absolute time difference between admission and culture collection
    valid_tests['time_diff'] = valid_tests['collect_datetime'] - valid_tests['admission_datetime']
    valid_tests['abs_time_diff'] = valid_tests['time_diff'].abs()

    # 6. For each unique episode, find the culture test with the minimum absolute time difference
    # Group by all original episode columns to preserve full metadata
    idx_min = valid_tests.groupby(list(ep_df.columns), dropna=False)['abs_time_diff'].idxmin()
    closest_tests = valid_tests.loc[idx_min].copy()

    # Format resulting time deltas
    # 5. Calculate and round time deltas
    closest_tests['time_to_culture_days'] = \
        (closest_tests['time_diff'].dt.total_seconds() / 86400.0).round(round_decimals)
    closest_tests['time_to_culture_hours'] = \
        (closest_tests['time_diff'].dt.total_seconds() / 3600.0).round(round_decimals)
    closest_tests['closest_culture_date'] = closest_tests['collect_datetime'].dt.date

    # Select columns to merge back
    cols_to_keep = list(ep_df.columns) + ['closest_culture_date', 'time_to_culture_days', 'time_to_culture_hours']
    result_df = closest_tests[cols_to_keep]

    # 7. Merge back to original episodes to ensure episodes with zero cultures are preserved
    final_episodes = ep_df.merge(result_df, on=list(ep_df.columns), how='left')

    # Drop temporary datetime helper columns from final output
    final_episodes.drop(columns=['admission_datetime', 'discharge_datetime'], inplace=True, errors='ignore')

    return final_episodes