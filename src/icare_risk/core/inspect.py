import pandas as pd
from typing import List


def print_patient_timeline(dataset, subject_id: int, table_name: str = 'vitals') -> None:
    """
    Visualizes the temporal boundaries of a single clinical table for a specific patient.

    This function queries the patient's hospital stays and provides a compact
    summary of event counts and edge timestamps (First/Last) for both historical
    and current events relative to each admission.

    Parameters
    ----------
    dataset : ClinicalDataset
        The initialized data provider engine.
    subject_id : int
        The unique identifier (SUBJECT) of the patient.
    table_name : str, optional
        The name of the clinical table to visualize (e.g., 'vitals').

    Returns
    -------
    None
        Prints the timeline directly to the console.
    """
    print(f"=== TIMELINE FOR PATIENT {subject_id} | TABLE: {table_name.upper()} ===")

    # 1. Fetch all hospital stays for this patient directly from the episodes view
    query = f"SELECT std_admission_time, std_discharge_time FROM episodes WHERE SUBJECT = {subject_id}"
    patient_stays = dataset.con.query(query).df()

    if patient_stays.empty:
        print("  Patient not found in episodes table.")
        return

    try:
        # 2. Fetch all current and historical data for this table
        curr = dataset.get_current_stay(table_name)
        hist = dataset.get_historical(table_name)
    except Exception as e:
        print(f"  Error fetching data for table '{table_name}': {e}")
        return

    # 3. Iterate through every known stay for this patient
    for _, stay in patient_stays.iterrows():
        adm = stay['std_admission_time']
        dis = stay['std_discharge_time']

        print(f"\nAdmission: {adm}  --->  Discharge: {dis}")

        # Filter the fetched data for THIS specific stay
        p_curr = curr[(curr['SUBJECT'] == subject_id) & (curr['std_admission_time'] == adm)]
        p_hist = hist[(hist['SUBJECT'] == subject_id) & (hist['std_admission_time'] == adm)]

        # Print Historical Summary
        if not p_hist.empty:
            print(
                f"  [HISTORICAL] {len(p_hist):<4} events. First: {p_hist['standard_time'].min()} | Last: {p_hist['standard_time'].max()}")
        else:
            print(f"  [HISTORICAL] No prior history found.")

        # Print Current Summary
        if not p_curr.empty:
            print(
                f"  [CURRENT]    {len(p_curr):<4} events. First: {p_curr['standard_time'].min()} | Last: {p_curr['standard_time'].max()}")
        else:
            print(f"  [CURRENT]    No events recorded during this specific stay.")

    print("\n" + "=" * 50)


def print_patient_multitable_timeline(dataset, subject_id: int, table_names: List[str]) -> None:
    """
    Succinctly visualizes the temporal boundaries across multiple clinical tables
    for a specific patient's hospital stays.

    This function groups a patient's data by hospital admission. Under each admission,
    it provides a compact summary of event counts and edge timestamps (First/Last) for
    both historical and current events across every requested table. This is ideal for
    quickly verifying cross-table data alignment and ensuring no data leakage occurred.

    Parameters
    ----------
    dataset : ClinicalDataset
        The initialized data provider engine.
    subject_id : int
        The unique identifier (SUBJECT) of the patient.
    table_names : list of str
        A list of table names to query (e.g., ['vitals', 'pathology', 'problems']).

    Returns
    -------
    None
        Prints the succinct multi-table timeline directly to the console.
    """
    print(f"=== MULTI-TABLE TIMELINE FOR PATIENT {subject_id} ===")

    # Find all unique hospital stays for this patient
    query = f"SELECT std_admission_time, std_discharge_time FROM episodes WHERE SUBJECT = {subject_id}"
    patient_stays = dataset.con.query(query).df()

    if patient_stays.empty:
        print("  Patient not found in episodes table.")
        return

    # Iterate through each unique hospital stay
    for _, stay in patient_stays.iterrows():
        adm = stay['std_admission_time']
        dis = stay['std_discharge_time']

        print(f"\n[STAY] {adm} to {dis}")

        # Query stats for each table requested
        for table in table_names:
            try:
                curr = dataset.get_current_stay(table)
                hist = dataset.get_historical(table)

                # Filter for this patient and this specific stay
                p_curr = curr[(curr['SUBJECT'] == subject_id) & (curr['std_admission_time'] == adm)]
                p_hist = hist[(hist['SUBJECT'] == subject_id) & (hist['std_admission_time'] == adm)]

                print(f"  > {table.upper()}:")

                # Print Historical Summary (Now showing First and Last)
                if not p_hist.empty:
                    print(
                        f"      Historical: {len(p_hist):<4} records (First: {p_hist['standard_time'].min()} | Last: {p_hist['standard_time'].max()})")
                else:
                    print(f"      Historical: 0    records")

                # Print Current Stay Summary (Showing First and Last)
                if not p_curr.empty:
                    print(
                        f"      Current:    {len(p_curr):<4} records (First: {p_curr['standard_time'].min()} | Last: {p_curr['standard_time'].max()})")
                else:
                    print(f"      Current:    0    records")

            except Exception as e:
                print(f"  > {table.upper()}: [Error fetching data: {str(e)}]")

    print("\n" + "=" * 50)


def print_df(df: pd.DataFrame, top: int = 5, threshold: int = 10) -> None:
  """Displays a DataFrame with shape and adaptive summary statistics

  for the last column based on its data type and cardinality.

  Parameters
  ----------
  df : pd.DataFrame
      The DataFrame to display.
  top : int, optional
      Number of top rows to display, by default 5.
  threshold : int, optional
      Cardinality threshold below which numeric/categorical columns
      show value counts instead of summary stats, by default 10.
  """
  last_col_name = ""
  summary_info = ""

  if not df.empty and len(df.columns) > 0:
    last_col = df.columns[-1]
    last_col_name = str(last_col)
    col_series = df[last_col]

    if pd.api.types.is_bool_dtype(col_series):
      true_count = col_series.sum()
      summary_info = f"True in '{last_col_name}': {true_count}"

    elif pd.api.types.is_numeric_dtype(col_series):
      if col_series.nunique() < threshold:
        counts = col_series.value_counts().to_dict()
        summary_info = f"Counts in '{last_col_name}': {counts}"
      else:
        summary_info = (
            f"Min: {col_series.min():.2g} | Max: {col_series.max():.2g} |"
            f" Median: {col_series.median():.2g} | Mean:"
            f" {col_series.mean():.2g}"
        )
    else:
      if col_series.nunique() < threshold:
        counts = col_series.value_counts().to_dict()
        summary_info = f"Counts in '{last_col_name}': {counts}"
      else:
        summary_info = f"Unique values: {col_series.nunique()}"

  print(f"\n=== Shape: {df.shape} | {summary_info} ===")
  print(df.head(top))