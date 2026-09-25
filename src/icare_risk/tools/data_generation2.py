"""
Dynamic Synthetic Data Generation
==================================
Single, consolidated pipeline for building the iCare synthetic tables from a
YAML config. This replaces the previous file which had accumulated several
duplicate/dead code paths (generate_custom_table, generate_custom_tablev2,
generate_eav_timeseries, generate_eav_timeseriesv2, a legacy `map_to`-based
schema for EAV tables, plus an unused `generate_clinical_data` sandbox).

"""

import random
import argparse

import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from datetime import datetime, timedelta


# =============================================================================
# SMALL HELPERS
# =============================================================================
def _safe_float(val, default, max_bound=10000.0):
    """Safely parse numbers, filtering out text artifacts or extreme database outliers."""
    try:
        if pd.isna(val):
            return default
        num = float(str(val))
        return max(-max_bound, min(num, max_bound))
    except (ValueError, TypeError):
        return default


def _resolve_lookups(table_config: dict) -> None:
    """Scans 'categorical_tuple' columns for lookup files and merges them with any
    manually defined tuples (used by 'relational' and 'entity' tables).

    Two separate concerns are now supported for lookup-backed tuples:

    ``columns``
        The **final** output column names (always required) - what you want
        the generated table to be called.
    ``lookup_columns`` (optional)
        The **source** column names to read from the lookup CSV, in the same
        order as ``columns``. Use this when the lookup file's headers don't
        match your desired output names (e.g. icareVIT.csv has a bare
        ``code`` column but you want the output called ``OBSERVATION_CODE``).
        If omitted, defaults to ``columns`` (i.e. the lookup file's headers
        are assumed to already be the names you want - the old behavior).

    For manually-specified tuples (no ``lookup_file``), ``values`` holds the
    literal rows. ``choices`` is accepted as a synonym for ``values``.
    """
    schema = table_config.get("schema", {})

    for col_name, col_def in schema.items():
        if col_def.get("type") == "categorical_tuple":

            # Allow "choices" as an alias for "values" (some tables, e.g.
            # ICARE_DEMOGRAPHIC_ANON's ETHNICITY_TUPLE/GENDER_TUPLE, use it)
            if "values" not in col_def and "choices" in col_def:
                col_def["values"] = col_def["choices"]
            if "values" not in col_def:
                col_def["values"] = []

            if "lookup_file" in col_def:
                file_path = Path(col_def["lookup_file"])
                if not file_path.exists():
                    raise FileNotFoundError(f"Lookup file not found: {file_path}")

                df_lookup = pd.read_csv(file_path)
                target_cols = col_def.get("columns", [])
                source_cols = col_def.get("lookup_columns", target_cols)

                if len(source_cols) != len(target_cols):
                    raise ValueError(
                        f"'{col_name}': 'lookup_columns' ({len(source_cols)} items) "
                        f"and 'columns' ({len(target_cols)} items) must be the same "
                        f"length and in matching order."
                    )

                missing_cols = [c for c in source_cols if c not in df_lookup.columns]
                if missing_cols:
                    raise ValueError(f"Lookup {file_path} missing columns: {missing_cols}")

                # Optional frequency-weighted sampling (e.g. icareVIT.csv / icareMIC.csv
                # 'total_ocurrences' column) instead of uniform random.choice.
                # Real lookup exports sometimes have blank/NaN counts for rare
                # rows - treat those as zero weight rather than crashing.
                weight_col = col_def.get("weight_col")
                if weight_col and weight_col in df_lookup.columns:
                    weights = df_lookup[weight_col].astype(float).fillna(0.0).values
                    total = weights.sum()
                    if total > 0:
                        col_def["_weights"] = (weights / total).tolist()
                    else:
                        print(f"  ⚠️ '{col_name}': all weights in '{weight_col}' are "
                              f"0/NaN - falling back to uniform sampling.")

                # Read using the SOURCE names, output using the TARGET names
                file_values = df_lookup[source_cols].values.tolist()
                col_def["values"].extend(file_values)

            if not col_def["values"]:
                raise ValueError(
                    f"Generation error on '{col_name}': You must provide either "
                    f"'values'/'choices' or a valid 'lookup_file' (or both)."
                )


def _load_eav_concepts(table_config: dict) -> dict:
    """Loads clinical concepts directly from an EAV lookup CSV.

    Every row of the CSV becomes a 'concept' (e.g. one vital sign or one lab
    test), with numeric bounds, units and metadata used later to populate the
    output table. This fully replaces the old YAML `clinical_concepts` +
    `source: clinical_concepts.xxx` indirection - the lookup CSV is now the
    single source of truth for concept metadata.
    """
    file_path = Path(table_config["lookup_file"])
    if not file_path.exists():
        raise FileNotFoundError(f"EAV lookup file not found: {file_path}")

    print(f"🔍 Loading EAV clinical concepts from lookup table: {file_path.name} ...")
    df_lookup = pd.read_csv(file_path)

    required_cols = ["code", "name", "unit"]
    missing = [c for c in required_cols if c not in df_lookup.columns]
    if missing:
        raise ValueError(f"EAV lookup file {file_path} is missing columns: {missing}")

    concepts_dict = {}
    for _, row in df_lookup.iterrows():
        code_str = str(row["code"])
        key = f"code_{code_str}"

        r_low = _safe_float(row.get("range_low"), 0.0)
        r_high = _safe_float(row.get("range_high"), 300.0)
        if r_low > r_high:
            r_low, r_high = r_high, r_low

        concepts_dict[key] = {
            "code": code_str,
            "name": str(row["name"]),
            "type": str(row.get("type", "float")),
            "unit": str(row["unit"]) if pd.notna(row["unit"]) else "none",
            "prob": float(row.get("prob", 0.20)) if pd.notna(row.get("prob")) else 0.20,
            "range": [r_low, r_high],
            "normal_low": _safe_float(row.get("normal_low"), np.nan),
            "normal_high": _safe_float(row.get("normal_high"), np.nan),
            "department": str(row.get("department", "Unspecified Dept")),
            "order_code": str(row.get("order_code", "UNK_ORD")),
            "order_name": str(row.get("order_name", "Unspecified Order")),
        }

    return concepts_dict


# =============================================================================
# TABLE GENERATORS
# =============================================================================
def generate_relational_table(unique_ids: list, table_config: dict, custom_tables: dict = None) -> pd.DataFrame:
    """Generates a relational table based on schema definitions, supporting weighted
    or uniform tuple picking."""
    rows_range = table_config.get('rows_per_patient_range', [1, 1])
    schema = table_config.get('schema', {})
    custom_tables = custom_tables or {}

    patient_col = []
    for pid in unique_ids:
        n_rows = np.random.randint(rows_range[0], rows_range[1] + 1)
        patient_col.extend([pid] * n_rows)

    total_rows = len(patient_col)
    if total_rows == 0:
        return pd.DataFrame(columns=schema.keys())

    df = pd.DataFrame()

    for col_name, props in schema.items():
        col_type = props.get('type')

        if col_type in ['primary_key', 'foreign_key']:
            source_table = props.get('source_table')
            if source_table and source_table in custom_tables:
                ref_df = custom_tables[source_table]
                ref_fk_col = [c for c in ref_df.columns if any(x in c for x in ['ID', 'IDENTIFIER', 'SUBJECT'])][0]
                ref_pat_col = [c for c in ref_df.columns if ref_df[c].isin(unique_ids).any()][0]
                enc_map = ref_df.groupby(ref_pat_col)[ref_fk_col].apply(list).to_dict()
                df[col_name] = [
                    np.random.choice(enc_map[pid]) if pid in enc_map and enc_map[pid] else np.nan
                    for pid in patient_col
                ]
            else:
                df[col_name] = patient_col

        elif col_type == 'date':
            start = pd.to_datetime(props['start'])
            end = pd.to_datetime(props['end'])
            span = max(0, (end - start).days)
            df[col_name] = [start + timedelta(days=random.randint(0, span)) for _ in range(total_rows)]
            df[col_name] = pd.to_datetime(df[col_name]).dt.date
            # Optional: blank out a proportion of rows (e.g. DEATH_DATE, where
            # most patients are still alive)
            null_prob = props.get('null_probability')
            if null_prob:
                mask = np.random.rand(total_rows) < null_prob
                df.loc[mask, col_name] = None

        elif col_type == 'date_offset':
            base_dates = pd.to_datetime(df[props['base_col']])
            offsets = [random.randint(props['days_range'][0], props['days_range'][1]) for _ in range(total_rows)]
            df[col_name] = (base_dates + pd.to_timedelta(offsets, unit='d')).dt.date

        elif col_type == 'categorical_tuple':
            choices = props['values']
            weights = props.get('_weights')
            if weights is not None and len(weights) == len(choices):
                selected_indices = np.random.choice(len(choices), size=total_rows, p=weights)
                selected = [choices[i] for i in selected_indices]
            else:
                selected = [random.choice(choices) for _ in range(total_rows)]
            for i, target_col in enumerate(props['columns']):
                df[target_col] = [item[i] for item in selected]

        elif col_type == 'int':
            df[col_name] = np.random.randint(props['range'][0], props['range'][1] + 1, size=total_rows)
        elif col_type == 'float':
            df[col_name] = np.round(np.random.uniform(props['range'][0], props['range'][1], size=total_rows), 2)
        elif col_type == 'enumerated':
            df[col_name] = [random.choice(props['values']) for _ in range(total_rows)]
        elif col_type == 'boolean':
            p_true = props.get('probability', 0.5)
            df[col_name] = np.random.choice([0, 1], size=total_rows, p=[1 - p_true, p_true])
        elif col_type == 'unique_id':
            df[col_name] = [random.randint(1000000, 9999999) for _ in range(total_rows)]
        elif col_type == 'time':
            df[col_name] = [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00" for _ in range(total_rows)]

    return df


def generate_eav_timeseries_table(unique_ids: list, custom_tables: dict, table_config: dict,
                                   freq: str, days: int, primary_key_col: str) -> pd.DataFrame:
    """Generates a long-format EAV timeseries table by sampling values within ranges
    defined in the lookup table.

    Reads a FLAT config (no `map_to`) - see module docstring for the list of
    recognized top-level keys (id_col, timestamp_col, value_col, concept_cols,
    extra_timestamp_cols, encounter, timestamp_offset, unique_id_col).
    """
    concepts = _load_eav_concepts(table_config)
    if not concepts:
        return pd.DataFrame()

    id_col = table_config.get('id_col', primary_key_col)
    timestamp_col = table_config.get('timestamp_col', 'TIMESTAMP')
    value_col = table_config.get('value_col', 'VALUE')
    unique_id_col = table_config.get('unique_id_col')
    concept_cols = table_config.get('concept_cols', {})
    extra_timestamp_cols = table_config.get('extra_timestamp_cols', [])
    encounter_cfg = table_config.get('encounter')
    offset_cfg = table_config.get('timestamp_offset')

    # Anchor patient timelines to their episode admission date, if available
    if 'ICARE_EPISODES_ANON' in custom_tables:
        base_df = custom_tables['ICARE_EPISODES_ANON']
        anchors = base_df.groupby('SUBJECT')['ADMISSION_DATE'].first().to_dict()
    else:
        anchors = {pid: pd.Timestamp('2024-01-01') for pid in unique_ids}

    steps_per_day = 24 // int(freq[:-1]) if freq[:-1].isdigit() else 6
    periods = days * steps_per_day

    all_rows = []
    for pid in unique_ids:
        start_dt = pd.to_datetime(anchors.get(pid, '2024-01-01'))
        timestamps = pd.date_range(start_dt, periods=periods, freq=freq)
        for ts in timestamps:
            for test_key in concepts.keys():
                all_rows.append({id_col: pid, 'timestamp': ts, 'test_key': test_key})

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Sample a numeric value within each concept's plausible range
    val_map = {k: v.get('range', [0.0, 1.0]) for k, v in concepts.items()}
    df['value'] = [
        round(float(np.random.uniform(val_map[k][0], val_map[k][1])), 2)
        if val_map[k][0] != val_map[k][1] else round(float(val_map[k][0]), 2)
        for k in df['test_key']
    ]

    # Randomly drop rows to simulate a test/observation not being taken
    probs = df['test_key'].map(lambda k: concepts[k].get('prob', 0.5))
    df = df[np.random.rand(len(df)) < probs].copy()

    # ---- Build the final, user-named output table -------------------------
    df_final = pd.DataFrame()
    df_final[id_col] = df[id_col]
    df_final[timestamp_col] = df['timestamp']
    df_final[value_col] = df['value']

    for extra_col in extra_timestamp_cols:
        df_final[extra_col] = df['timestamp']

    for concept_field, out_col in concept_cols.items():
        df_final[out_col] = df['test_key'].map(lambda k: concepts[k].get(concept_field, np.nan))

    if unique_id_col:
        df_final[unique_id_col] = [random.randint(1000000, 9999999) for _ in range(len(df))]

    if encounter_cfg:
        enc_col = encounter_cfg.get('col')
        source_table = encounter_cfg.get('source_table')
        if enc_col and source_table in custom_tables:
            enc_df = custom_tables[source_table]
            enc_map = enc_df.groupby('SUBJECT')['ENCNTR_ID'].first().to_dict()
            df_final[enc_col] = df[id_col].map(enc_map)

    if offset_cfg:
        off_col = offset_cfg['col']
        base_col_name = offset_cfg.get('base_col')
        hrs = offset_cfg.get('hours_range', [1, 24])
        base_series = df_final[base_col_name] if base_col_name in df_final.columns else df['timestamp']
        offsets = pd.to_timedelta(np.random.randint(hrs[0], hrs[1] + 1, size=len(df)), unit='h')
        df_final[off_col] = base_series + offsets

    return df_final.sort_values([id_col, timestamp_col]).reset_index(drop=True)


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================
def generate_synthetic_cohort(config_path: str, output_dir: str = None) -> dict:
    """Generates synthetic data from a YAML config file and returns the tables."""

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[3]
        date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out_path = project_root / 'data' / 'synthetic' / date_str
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 TARGET SAVE DIRECTORY: {out_path.absolute()}\n")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file.absolute()}")

    print(f"⚙️ Loading standalone config from: {config_file.resolve()}")
    with open(config_file, 'r') as f:
        data_config = yaml.safe_load(f)

    params = data_config.get('generation_params', {})
    n_patients = params.get('n_patients', 100)
    unique_ids = list(range(10001, 10001 + n_patients))
    primary_key_col = params.get('patient_col', 'SUBJECT')

    custom_tables = {}

    if 'tables' in data_config:
        print(f"Generating configured tables for {n_patients} patients...")

        # NOTE: The YAML must list Parent tables (e.g. ICARE_EPISODES_ANON)
        # before Child tables (e.g. Pharmacy, Vitals) so foreign keys exist
        # when needed. Dict order in the YAML == generation order.
        for table_name, table_config in data_config['tables'].items():
            if not isinstance(table_config, dict) or 'type' not in table_config:
                print(f"  ⚠️ Skipping '{table_name}': Missing 'type' in YAML.")
                continue

            print(f" -> Building {table_name} [{table_config['type']}]...")

            table_type = table_config['type']
            if table_type == 'relational':
                _resolve_lookups(table_config)
                df_table = generate_relational_table(
                    unique_ids=unique_ids,
                    table_config=table_config,
                    custom_tables=custom_tables
                )

            elif table_type == 'eav_timeseries':
                df_table = generate_eav_timeseries_table(
                    unique_ids=unique_ids,
                    custom_tables=custom_tables,
                    table_config=table_config,
                    freq=params.get('freq', '4h'),
                    days=params.get('days', 10),
                    primary_key_col=primary_key_col
                )
            else:
                print(f"  ⚠️ Unknown table type: {table_type}")
                continue

            custom_tables[table_name] = df_table

            file_name = f"{table_name.lower()}.csv"
            df_table.to_csv(out_path / file_name, index=False)

    print(f"\n✨ Cohort generation complete! Saved to {out_path.absolute()}")
    return custom_tables


def main():
    parser = argparse.ArgumentParser(description="Dynamic Synthetic Data Generation")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to the YAML config file. Falls back to the packaged '
             "'config/icare/generate_db.yaml' (via icare_risk.utils.io.load_pkg_yaml) "
             'if omitted.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save the generated CSVs (defaults to a timestamped folder)'
    )
    args = parser.parse_args()

    print("==================================================")
    print("🧬 Dynamic Synthetic Data Generation")
    print("==================================================")

    config_path = args.config
    if config_path is None:
        # Fallback for in-package usage: resolve the packaged default config
        # to a real file path so generate_synthetic_cohort can load it the
        # same way regardless of how it was invoked.
        try:
            import icare_risk
            config_path = str(Path(icare_risk.__file__).parent / 'config' / 'icare' / 'generate_db.yaml')
        except ImportError:
            parser.error("--config is required when 'icare_risk' package is not installed.")

    generate_synthetic_cohort(config_path=config_path, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
