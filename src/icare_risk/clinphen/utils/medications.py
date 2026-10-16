import pandas as pd


def get_qualifying_last_abx(
    df_episodes: pd.DataFrame,
    df_admin: pd.DataFrame,
    target_abx: list[str],
    subject_col: str = "SUBJECT",
    episode_date_col: str = "ADMISSION_DATE",
    episode_time_col: str = "ADMISSION_TIME",
    admin_dt_col: str = "ADMINISTRATION_DT",
    order_id_col: str = "TEMPLATE_OIRDED_ID",
    med_name_col: str = "MEDICATION_NAME_SHORT",
    admin_status_col: str = "ADMINISTRATION_STATUS",
    given_status_val: str = "Given",
    min_duration_hours: float = 24.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Identifies if the absolute last medication given before an episode was a

    target antibiotic administered for more than a specified duration (default
    >24h).
    """
    df_episodes_proc = df_episodes.copy()

    # 1. Construct unified episode event datetime timestamp
    if episode_time_col and episode_time_col in df_episodes_proc.columns:
        combined_ts = (
            df_episodes_proc[episode_date_col].astype(str)
            + " "
            + df_episodes_proc[episode_time_col].astype(str)
        )
        df_episodes_proc["_event_datetime"] = pd.to_datetime(combined_ts)
    else:
        df_episodes_proc["_event_datetime"] = pd.to_datetime(
            df_episodes_proc[episode_date_col]
        )

    # 2. Filter administration records to given doses and convert timestamp
    admin_mask = (
        df_admin[admin_status_col].astype(str).str.strip().str.lower()
        == given_status_val.lower()
    )
    df_admin_given = df_admin[admin_mask].copy()
    df_admin_given["_admin_datetime"] = pd.to_datetime(
        df_admin_given[admin_dt_col]
    )

    # 3. Group by subject and order ID to calculate course durations
    df_courses = (
        df_admin_given.groupby([subject_col, order_id_col, med_name_col])
        .agg(
            _first_admin=("_admin_datetime", "min"),
            _last_admin=("_admin_datetime", "max"),
        )
        .reset_index()
    )

    df_courses["_abx_duration_hours"] = (
        df_courses["_last_admin"] - df_courses["_first_admin"]
    ).dt.total_seconds() / 3600.0

    # 4. Sort chronologically for merge_asof
    df_courses = df_courses.sort_values("_last_admin")
    df_episodes_proc = df_episodes_proc.sort_values("_event_datetime")

    # 5. Merge nearest prior medication course before the episode timestamp
    df_merged = pd.merge_asof(
        df_episodes_proc,
        df_courses[
            [subject_col, "_last_admin", med_name_col, "_abx_duration_hours"]
        ],
        left_on="_event_datetime",
        right_on="_last_admin",
        by=subject_col,
        direction="backward",
    )

    # 6. Apply qualification criteria (target list AND duration threshold)
    target_abx_lower = [abx.lower() for abx in target_abx]
    is_target = (
        df_merged[med_name_col].astype(str).str.lower().isin(target_abx_lower)
    )
    is_gt_min_duration = df_merged["_abx_duration_hours"] > min_duration_hours

    df_merged["qualifying_last_abx"] = df_merged[med_name_col].where(
        is_target & is_gt_min_duration
    )

    # 7. Print summary if verbose is enabled
    if verbose:
        qual_count = df_merged["qualifying_last_abx"].notna().sum()
        total_events = len(df_merged)
        pct = (qual_count / total_events * 100) if total_events > 0 else 0.0

        print(
            f"--- Qualifying Last Antibiotic Summary (> {min_duration_hours:.0f}h) ---"
        )
        print(
            f"  Qualifying ABX Events : {qual_count:>7,} / {total_events:,} ({pct:.1f}%)\n"
        )

        counts = df_merged["qualifying_last_abx"].value_counts(dropna=True)
        if not counts.empty:
            max_name_len = max(len(str(name)) for name in counts.index)
            for abx_name, count in counts.items():
                print(f"  {abx_name:<{max_name_len}} : {count:>7,} events")
        else:
            print("  No qualifying antibiotics identified.")
        print("-" * 50)

    # 8. Clean up intermediate processing columns
    cols_to_drop = [
        "_event_datetime",
        "_last_admin",
        med_name_col,
        "_abx_duration_hours",
    ]
    if med_name_col in df_episodes.columns:
        cols_to_drop.remove(med_name_col)

    return df_merged.drop(
        columns=[c for c in cols_to_drop if c in df_merged.columns]
    )