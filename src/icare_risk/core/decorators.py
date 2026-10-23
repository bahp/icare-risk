import functools
import pandas as pd
from typing import Union, Dict, List


def clinical_feature(
        output_name: str,
        tables: Union[str, Dict[str, List[str]]],
        agg_func: str = 'max'
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(dataset, *args, **kwargs):
            # Dynamically fetch the primary keys from the dataset
            subj_col = getattr(dataset, 'subject_col', 'SUBJECT')
            adm_col = getattr(dataset, 'admission_col', 'std_admission_time')

            dfs = {}
            if isinstance(tables, str):
                table_dict = {tables: None}
            elif isinstance(tables, list):
                table_dict = {t: None for t in tables}
            else:
                table_dict = tables

            for table_name, columns in table_dict.items():
                try:
                    dfs[table_name] = dataset.get_current_stay(table_name, columns=columns)
                except TypeError:
                    df = dataset.get_current_stay(table_name)
                    if columns and not df.empty:
                        # Use the dynamic columns here
                        safe_cols = list(set(columns + [subj_col, adm_col]))
                        existing_cols = [c for c in safe_cols if c in df.columns]
                        dfs[table_name] = df[existing_cols]
                    else:
                        dfs[table_name] = df

            if isinstance(tables, str):
                scored_df = func(dfs[tables], *args, **kwargs)
            else:
                scored_df = func(dfs, *args, **kwargs)

            if output_name in scored_df.columns:
                # Group by the dynamic columns
                grouped = scored_df.groupby([subj_col, adm_col])[output_name].agg(agg_func)
                return grouped.reset_index()

            return scored_df
        return wrapper
    return decorator