import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

from icare_risk.utils import load_yaml_config

import pandas as pd
from pathlib import Path


def _safe_float(val, default, max_bound=10000.0):
    """Safely parse numbers, filtering out text artifacts or extreme database outliers."""
    try:
        if pd.isna(val):
            return default
        #val_str = str(val).replace(',', '').strip()
        num = float(str(val))
        return max(-max_bound, min(num, max_bound))
    except (ValueError, TypeError):
        return default

def _resolve_eav_lookups(table_config: dict, data_config: dict) -> None:
    """Reads an EAV CSV lookup file and merges it with any manual YAML overrides."""
    if table_config.get("type") != "eav_timeseries" or "lookup_file" not in table_config:
        return

    file_path = Path(table_config["lookup_file"])
    if not file_path.exists():
        raise FileNotFoundError(f"EAV lookup file not found: {file_path}")

    print(f"🔍 Loading clinical concepts from lookup table: {file_path.name} ...")
    df_lookup = pd.read_csv(file_path)

    required_cols = ["code", "name", "unit"]
    missing = [c for c in required_cols if c not in df_lookup.columns]
    if missing:
        raise ValueError(f"EAV lookup file {file_path} is missing columns: {missing}")

    concepts_dict = {}
    for _, row in df_lookup.iterrows():
        code_str = str(row["code"])
        key = f"code_{code_str}"

        # Build concept entirely from CSV row, with safe fallbacks for missing metadata
        concepts_dict[key] = {
            "code": code_str,
            "name": str(row["name"]),
            "type": str(row.get("type", "float")),
            "unit": str(row["unit"]) if pd.notna(row["unit"]) else "none",
            "prob": float(row.get("prob", 0.20)),
            "range": [float(row.get("range_low", 0.0)), float(row.get("range_high", 300.0))],
            "normal_low": float(row["normal_low"]) if "normal_low" in row and pd.notna(row["normal_low"]) else np.nan,
            "normal_high": float(row["normal_high"]) if "normal_high" in row and pd.notna(
                row["normal_high"]) else np.nan,
            "department": str(row.get("department", "Unspecified Dept")),
            "order_code": str(row.get("order_code", "UNK_ORD")),
            "order_name": str(row.get("order_name", "Unspecified Order")),
        }

    concept_key = f"eav_loaded_{id(table_config)}"
    if "clinical_concepts" not in data_config:
        data_config["clinical_concepts"] = {}

    data_config["clinical_concepts"][concept_key] = concepts_dict
    table_config["source"] = f"clinical_concepts.{concept_key}"

def _resolve_lookups(table_config: dict) -> None:
    """Scans for lookup files and merges them with any manually defined tuples.

    .. note: We could add weight col, and use total_ocurrences.
    """
    schema = table_config.get("schema", {})

    for col_name, col_def in schema.items():
        if col_def.get("type") == "categorical_tuple":

            # 1. Initialize the values list if it wasn't manually defined in YAML
            if "values" not in col_def:
                col_def["values"] = []

            # 2. If a lookup file is specified, load it and append to the list
            if "lookup_file" in col_def:
                file_path = Path(col_def["lookup_file"])
                if not file_path.exists():
                    raise FileNotFoundError(f"Lookup file not found: {file_path}")

                df_lookup = pd.read_csv(file_path)
                target_cols = col_def.get("columns", [])

                # Verify all requested columns exist in the CSV
                missing_cols = [c for c in target_cols if c not in df_lookup.columns]
                if missing_cols:
                    raise ValueError(f"Lookup {file_path} missing columns: {missing_cols}")

                # Extract as list of lists and append to existing values
                file_values = df_lookup[target_cols].values.tolist()
                col_def["values"].extend(file_values)

            # 3. Final validation to ensure we have data to generate from
            if not col_def["values"]:
                raise ValueError(
                    f"Generation error on '{col_name}': You must provide either "
                    f"'values' or a valid 'lookup_file' (or both)."
                )


def _load_eav_concepts(table_config: dict) -> dict:
    """
    Loads clinical concepts directly from an EAV lookup CSV (Mode B: Range-based generation).
    Each row defines a concept with numeric bounds, units, and metadata.
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

def generate_synthetic_cohort(config_path: str, output_dir: str = None) -> dict:
    """Generates synthetic data from a YAML config file and returns the tables."""

    # 1. Setup target directory
    if output_dir is None:
        date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out_path = 'app' / 'data' / 'synthetic' / date_str
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 TARGET SAVE DIRECTORY: {out_path.absolute()}\n")

    # 2. Load Configuration securely
    # Assuming load_yaml_config handles the fallback to default config names
    # 2. Load Configuration directly from your target YAML file (Standalone)
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file.absolute()}")

    print(f"⚙️ Loading standalone config from: {config_file.resolve()}")
    with open(config_file, 'r') as f:
        data_config = yaml.safe_load(f)

    # 3. Establish Global Parameters
    params = data_config.get('generation_params', {})
    n_patients = params.get('n_patients', 100)
    unique_ids = list(range(10001, 10001 + n_patients))
    primary_key_col = params.get('patient_col', 'SUBJECT')

    custom_tables = {}

    # 4. Generate Configured Tables
    if 'tables' in data_config:
        print(f"Generating configured tables for {n_patients} patients...")

        for table_name, table_config in data_config['tables'].items():
            if not isinstance(table_config, dict) or 'type' not in table_config:
                print(f"  ⚠️ Skipping '{table_name}': Missing 'type' in YAML.")
                continue

            print(f" -> Building {table_name} [{table_config['type']}]...")

            """
            # --- PRE-PROCESS LOOKUPS HERE ---
            _resolve_lookups(table_config)
            _resolve_eav_lookups(table_config, data_config)
            # --------------------------------

            if table_config['type'] == 'relational':
                df_table = generate_custom_table(
                    unique_ids=unique_ids,
                    table_config=table_config,
                    custom_tables=custom_tables
                )

            elif table_config['type'] == 'eav_timeseries':
                df_table = generate_eav_timeseries(
                    unique_ids=unique_ids,
                    custom_tables=custom_tables,
                    table_config=table_config,
                    clinical_concepts=data_config,
                    freq=params.get('freq', '4h'),
                    days=params.get('days', 10),
                    primary_key_col=primary_key_col
                )
            """
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

            # Save output to CSV
            file_name = f"{table_name.lower()}.csv"
            df_table.to_csv(out_path / file_name, index=False)

    print(f"\n✨ Cohort generation complete! Saved to {out_path.absolute()}")
    return custom_tables

# ----------------------------------------------------------------------------------
# Helper methods
# ----------------------------------------------------------------------------------
def generate_clinical_data(config,
                           n_patients,
                           days=6,
                           freq='8h',
                           output_format='long',
                           icd_map={},
                           id_col_name='patient_id'):
    """Generates all data dynamically based on the provided schema configuration."""
    # Create unique ids
    unique_ids = np.arange(101, 101 + n_patients)

    # Generate demographics data using the schema
    df_static = generate_demographics(
        unique_ids=unique_ids,
        demo_schema=config.get('demographics_schema', {}),
        conditions=config.get('conditions', []),
        icd_map=icd_map,
        id_col_name=id_col_name
    )

    # Generate timeseries data
    df_ts = generate_ts(
        unique_ids=unique_ids,
        df_static=df_static,
        ts_config=config.get('ts_config', {}),
        days=days,
        freq=freq,
        output_format=output_format
    )

    return df_static, df_ts


def generate_demographics(unique_ids, demo_schema, conditions,
                          icd_map, id_col_name='patient_id'):
    """Generates static demographic data strictly based on the schema config."""
    n = len(unique_ids)
    df = pd.DataFrame({id_col_name: unique_ids})

    # 1. Generate based on schema definition
    for col, props in demo_schema.items():
        if props['type'] == 'int':
            df[col] = np.random.randint(props['range'][0], props['range'][1] + 1, size=n)
        elif props['type'] == 'float':
            # Rounding to 1 decimal place for realistic weights/measurements
            df[col] = np.round(np.random.uniform(props['range'][0], props['range'][1], size=n), 1)
        elif props['type'] == 'enumerated':
            df[col] = np.random.choice(props['values'], size=n)

    # 2. Generate Binary Conditions (Comorbidities)
    if isinstance(conditions, dict):
        for cond, prob in conditions.items():
            df[cond] = np.random.choice([0, 1], size=n, p=[1 - prob, prob])

    # 3. Logic-Driven ICD-10 Code Generation
    # We create a string of codes for each patient based on their assigned conditions
    def map_to_icd(row, condition_name):
        if row[condition_name] == 1 and condition_name in icd_map:
            return np.random.choice(icd_map[condition_name])
        return "N/A"

    # Apply mapping only for conditions that exist in the ICD map
    for condition in icd_map.keys():
        if condition in df.columns:
            df[f'{condition}_icd'] = df.apply(lambda r: map_to_icd(r, condition), axis=1)

    # 3. Risk score & label (You can externalize these weights to config later if desired)
    risk_score = (
            (df.get('age', 50) * 0.5) +
            (df.get('diabetes', 0) * 20) +
            (df.get('CKD', 0) * 15)
    )
    df['risk'] = risk_score
    df['sepsis_case'] = (risk_score > np.percentile(risk_score, 70)).astype(int)

    return df


def generate_ts(unique_ids, df_static, ts_config,
                days=6, freq='8h', output_format='long',
                id_col_name='patient_id'):
    """Generates clinical data with LEARNABLE patterns, bounded by schema ranges."""
    # 1. Setup Time Grid
    periods = days * (24 // int(freq[:-1]))
    timestamps = pd.date_range('2024-01-01', periods=periods, freq=freq)
    idx = pd.MultiIndex.from_product(
        [unique_ids, timestamps], names=[id_col_name, 'date']
    )
    df = pd.DataFrame(index=idx).reset_index()

    # 2. Determine Sepsis Onset
    septic_ids = df_static[df_static['sepsis_case'] == 1][id_col_name].values
    onset_map = {pid: np.random.randint(5, periods - 5) for pid in septic_ids}

    df['step'] = df.groupby(id_col_name).cumcount()
    df['onset_step'] = df[id_col_name].map(onset_map)
    df['is_sick'] = (df.step > df.onset_step).astype(int)

    # 3. Generate Learnable Signals
    t = df['step'].values
    steps_per_day = 24 // int(freq[:-1])

    for k, props in ts_config.items():
        # A. BASELINE: deterministic hash-like
        patient_base = df[id_col_name].map(lambda x: (x % 50) + 50)

        # B. PATTERN: Sine Wave (Circadian Rhythm)
        amp = 10 if k in ['hr', 'temp'] else 2
        seasonality = amp * np.sin(2 * np.pi * t / steps_per_day)

        # C. RANDOM WALK (Physiology is smooth)
        noise = np.random.normal(0, 1, size=len(df))
        smooth_noise = np.convolve(noise, np.ones(3) / 3, mode='same') * 5

        # D. SEPSIS DRIFT
        drift_intensity = 2.0 if k in ['hr', 'temp', 'lactate'] else 0.5
        drift = np.where(df['is_sick'], (df['step'] - df['onset_step']) * drift_intensity, 0)

        # Combine
        df[k] = patient_base + seasonality + smooth_noise + drift

        # E. SCHEMA BOUNDARIES (Replaces the hardcoded clip)
        if 'range' in props:
            df[k] = df[k].clip(lower=props['range'][0], upper=props['range'][1])

        # Optional: round floats to a sensible precision
        if props.get('type') == 'float':
            df[k] = np.round(df[k], 2)

    if output_format == 'tidy':
        return df.set_index([id_col_name, 'step'])

    # Melt logic to long format
    df = df.melt(id_vars=[id_col_name, 'date'], var_name='test', value_name='result')
    unit_map = {k: v.get('unit', '') for k, v in ts_config.items()}
    df['unit'] = df['test'].map(unit_map)

    return df.sort_values([id_col_name, 'date']).reset_index(drop=True)


def apply_missingness(df, ts_config, default_rate=0.5):
    """Randomly removes rows to simulate missing data based on config probabilities."""
    df_out = df.copy()
    target_cols = [c for c in df.columns if c in ts_config]

    # Extract prob from config, fallback to default_rate if missing
    probs = np.array([ts_config[c].get('prob', default_rate) for c in target_cols])

    random_matrix = np.random.rand(len(df), len(target_cols))
    mask_matrix = random_matrix > probs
    df_out[target_cols] = df_out[target_cols].mask(mask_matrix)
    return df_out


import random
from datetime import timedelta

# src/generators.py

import pandas as pd
import numpy as np
import random

import pandas as pd
import numpy as np
import random

import pandas as pd
import numpy as np
import random
from datetime import timedelta

# REVISIT
def _load_eav_concepts(table_config: dict) -> dict:
    """
    Loads clinical concepts directly from an EAV lookup CSV (Mode B: Range-based generation).
    Each row defines a concept with numeric bounds, units, and metadata.
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


def generate_relational_table(unique_ids: list, table_config: dict, custom_tables: dict = None) -> pd.DataFrame:
    """Generates a relational table based on schema definitions, supporting weighted or uniform tuple picking."""
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
            df[col_name] = [start + timedelta(days=random.randint(0, max(0, (end - start).days))) for _ in range(total_rows)]
            df[col_name] = pd.to_datetime(df[col_name]).dt.date

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
            df[col_name] = np.random.choice([0, 1], size=total_rows,
                                            p=[1 - props.get('probability', 0.5), props.get('probability', 0.5)])
        elif col_type == 'unique_id':
            df[col_name] = [random.randint(1000000, 9999999) for _ in range(total_rows)]
        elif col_type == 'time':
            df[col_name] = [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00" for _ in range(total_rows)]

    return df


def generate_eav_timeseries_table(unique_ids: list, custom_tables: dict, table_config: dict, freq: str, days: int, primary_key_col: str) -> pd.DataFrame:
    """Generates a long-format EAV timeseries table by sampling values within ranges defined in the lookup table."""
    schema = table_config.get('schema', {})
    concepts = _load_eav_concepts(table_config)

    if not concepts:
        return pd.DataFrame()

    if 'ICARE_EPISODES_ANON' in custom_tables:
        base_df = custom_tables['ICARE_EPISODES_ANON']
        anchors = base_df.groupby('SUBJECT')['ADMISSION_DATE'].first().to_dict()
    else:
        anchors = {pid: pd.Timestamp('2024-01-01') for pid in unique_ids}

    all_rows = []
    periods = days * (24 // int(freq[:-1] if freq[:-1].isdigit() else 1))

    for pid in unique_ids:
        start_dt = pd.to_datetime(anchors.get(pid, '2024-01-01'))
        timestamps = pd.date_range(start_dt, periods=periods, freq=freq)

        for ts in timestamps:
            for test_key in concepts.keys():
                all_rows.append({
                    primary_key_col: pid,
                    'timestamp': ts,
                    'test_key': test_key
                })

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Range-based value generation (Mode B)
    val_map = {k: v.get('range', [0.0, 1.0]) for k, v in concepts.items()}
    df['value'] = [
        round(float(np.random.uniform(val_map[k][0], val_map[k][1])), 2)
        if val_map[k][0] != val_map[k][1] else round(float(val_map[k][0]), 2)
        for k in df['test_key']
    ]

    # Apply Missingness probability
    probs = df['test_key'].map(lambda k: concepts[k].get('prob', 0.5))
    df = df[np.random.rand(len(df)) < probs].copy()

    # Map concept metadata
    df['concept.code'] = df['test_key'].map(lambda k: concepts[k].get('code', 'UNK'))
    df['concept.name'] = df['test_key'].map(lambda k: concepts[k].get('name', 'Unknown'))
    df['concept.unit'] = df['test_key'].map(lambda k: concepts[k].get('unit', ''))
    df['concept.department'] = df['test_key'].map(lambda k: concepts[k].get('department', 'Unspecified'))
    df['concept.order_code'] = df['test_key'].map(lambda k: concepts[k].get('order_code', 'UNK_ORD'))
    df['concept.order_name'] = df['test_key'].map(lambda k: concepts[k].get('order_name', 'Unspecified Order'))
    df['concept.normal_low'] = df['test_key'].map(lambda k: concepts[k].get('normal_low', np.nan))
    df['concept.normal_high'] = df['test_key'].map(lambda k: concepts[k].get('normal_high', np.nan))

    df_final = pd.DataFrame()
    for col_name, rules in schema.items():
        target = rules.get('map_to')
        if target == primary_key_col:
            df_final[col_name] = df[primary_key_col]
        elif target == 'timestamp':
            df_final[col_name] = df['timestamp']
        elif target == 'value':
            df_final[col_name] = df['value']
        elif target == 'unique_id':
            df_final[col_name] = [random.randint(1000000, 9999999) for _ in range(len(df))]
        elif target == 'timestamp_offset':
            base = df['timestamp']
            df_final[col_name] = base + pd.to_timedelta(
                np.random.randint(rules['hours_range'][0], rules['hours_range'][1] + 1), unit='h')
        elif target and target.startswith('concept.'):
            df_final[col_name] = df[target]
        elif target == 'foreign_key' and rules.get('source_table') in custom_tables:
            enc_df = custom_tables[rules['source_table']]
            enc_map = enc_df.groupby('SUBJECT')['ENCNTR_ID'].first().to_dict()
            df_final[col_name] = df[primary_key_col].map(enc_map)

    sort_cols = [primary_key_col]
    time_cols = [k for k, v in schema.items() if v.get('map_to') == 'timestamp']
    if time_cols:
        sort_cols.append(time_cols[0])

    return df_final.sort_values(sort_cols).reset_index(drop=True)

def generate_custom_table(unique_ids, table_config, custom_tables=None):
    """Generates a relational table based on a dynamic YAML schema definition."""
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
                # Dynamic ID lookup
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
            df[col_name] = [start + timedelta(days=random.randint(0, (end - start).days)) for _ in range(total_rows)]
            df[col_name] = pd.to_datetime(df[col_name]).dt.date

        elif col_type == 'date_offset':
            base_dates = pd.to_datetime(df[props['base_col']])
            offsets = [random.randint(props['days_range'][0], props['days_range'][1]) for _ in range(total_rows)]
            df[col_name] = (base_dates + pd.to_timedelta(offsets, unit='d')).dt.date

        elif col_type == 'categorical_tuple':
            choices = props['values']
            selected = [random.choice(choices) for _ in range(total_rows)]
            for i, target_col in enumerate(props['columns']):
                df[target_col] = [item[i] for item in selected]

        # Add remaining simple types (int, float, enumerated, boolean, unique_id)
        elif col_type == 'int':
            df[col_name] = np.random.randint(props['range'][0], props['range'][1] + 1, size=total_rows)
        elif col_type == 'float':
            df[col_name] = np.round(np.random.uniform(props['range'][0], props['range'][1], size=total_rows), 2)
        elif col_type == 'enumerated':
            df[col_name] = [random.choice(props['values']) for _ in range(total_rows)]
        elif col_type == 'boolean':
            df[col_name] = np.random.choice([0, 1], size=total_rows,
                                            p=[1 - props.get('probability', 0.5), props.get('probability', 0.5)])
        elif col_type == 'unique_id':
            df[col_name] = [random.randint(1000000, 9999999) for _ in range(total_rows)]
        elif col_type == 'time':
            df[col_name] = [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00" for _ in range(total_rows)]

    return df


def generate_eav_timeseries(unique_ids, custom_tables, table_config, clinical_concepts, freq, days, primary_key_col):
    """Generates a long-format Entity-Attribute-Value (EAV) table anchored to Episode dates."""
    schema = table_config.get('schema', {})

    # Path Resolution for clinical_concepts
    source_path = table_config.get('source', '').split('.')
    concepts = clinical_concepts
    for key in source_path:
        if key in concepts:
            concepts = concepts[key]
        else:
            print(f"  ❌ Error: Clinical concept path '{key}' not found in configuration.")
            return pd.DataFrame()

    # Step 1: Anchor to Episodes if available, otherwise use default range
    if 'ICARE_EPISODES_ANON' in custom_tables:
        base_df = custom_tables['ICARE_EPISODES_ANON']
        # Use first episode per patient as anchor
        anchors = base_df.groupby('SUBJECT')['ADMISSION_DATE'].first().to_dict()
    else:
        anchors = {pid: pd.Timestamp('2024-01-01') for pid in unique_ids}

    # Step 2: Build Time Grid per Patient
    all_rows = []
    periods = days * (24 // int(freq[:-1] if freq[:-1].isdigit() else 1))

    for pid in unique_ids:
        start_dt = pd.to_datetime(anchors.get(pid, '2024-01-01'))
        timestamps = pd.date_range(start_dt, periods=periods, freq=freq)

        for ts in timestamps:
            for test_key in concepts.keys():
                all_rows.append({
                    primary_key_col: pid,
                    'timestamp': ts,
                    'test_key': test_key
                })

    df = pd.DataFrame(all_rows)
    if df.empty: return df

    # Step 3: Map Values and Metadata
    # Each concept is typed ("float" -> sample from its numeric range,
    # "enumerated" -> sample one of its pipe-delimited string_values options).
    def _sample_value(concept: dict):
        data_type = str(concept.get('type', 'float')).lower()
        if data_type == 'enumerated':
            options = concept.get('options') or []
            if options:
                return random.choice(options)
            return concept.get('name', 'Unknown')  # safe fallback if no options parsed
        low, high = concept.get('range', [0.0, 1.0])
        if low > high:
            low, high = high, low
        if low == high:
            return round(float(low), 2)
        return round(float(np.random.uniform(low, high)), 2)

    df['value'] = [_sample_value(concepts[k]) for k in df['test_key']]

    # Apply Missingness
    probs = df['test_key'].map(lambda k: concepts[k].get('prob', 0.5))
    df = df[np.random.rand(len(df)) < probs].copy()

    # Apply Metadata
    df['concept.code'] = df['test_key'].map(lambda k: concepts[k].get('code', 'UNK'))
    df['concept.name'] = df['test_key'].map(lambda k: concepts[k].get('name', 'Unknown'))
    df['concept.unit'] = df['test_key'].map(lambda k: concepts[k].get('unit', ''))
    df['concept.department'] = df['test_key'].map(lambda k: concepts[k].get('department', 'Unspecified'))
    df['concept.order_code'] = df['test_key'].map(lambda k: concepts[k].get('order_code', 'UNK_ORD'))
    df['concept.order_name'] = df['test_key'].map(lambda k: concepts[k].get('order_name', 'Unspecified Order'))
    # Normal ranges (default to generic if missing)
    df['concept.normal_low'] = df['test_key'].map(lambda k: concepts[k].get('normal_low', np.nan))
    df['concept.normal_high'] = df['test_key'].map(lambda k: concepts[k].get('normal_high', np.nan))

    # Step 4: Map to Final Schema
    df_final = pd.DataFrame()
    for col_name, rules in schema.items():
        target = rules.get('map_to')
        if target == primary_key_col:
            df_final[col_name] = df[primary_key_col]
        elif target == 'timestamp':
            df_final[col_name] = df['timestamp']
        elif target == 'value':
            df_final[col_name] = df['value']
        elif target == 'unique_id':
            df_final[col_name] = [random.randint(1000000, 9999999) for _ in range(len(df))]
        elif target == 'timestamp_offset':
            base = df['timestamp']  # Use internal grid timestamp
            df_final[col_name] = base + pd.to_timedelta(
                np.random.randint(rules['hours_range'][0], rules['hours_range'][1]), unit='h')
        elif target and target.startswith('concept.'):
            df_final[col_name] = df[target]
        elif target == 'foreign_key' and rules.get('source_table') in custom_tables:
            # Simple link to ENCNTR_ID for vitals
            enc_df = custom_tables[rules['source_table']]
            enc_map = enc_df.groupby('SUBJECT')['ENCNTR_ID'].first().to_dict()
            df_final[col_name] = df[primary_key_col].map(enc_map)

    return df_final.sort_values([primary_key_col, list(df_final.columns)[2]])

def generate_custom_tablev2(unique_ids, table_config, custom_tables=None):
    """Generates a relational table based on a dynamic YAML schema definition."""
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
                ref_fk_col = [c for c in ref_df.columns if 'ID' in c or 'IDENTIFIER' in c][0]
                ref_pat_col = [c for c in ref_df.columns if ref_df[c].isin(unique_ids).any()][0]
                enc_map = ref_df.groupby(ref_pat_col)[ref_fk_col].apply(list).to_dict()
                df[col_name] = [
                    np.random.choice(enc_map[pid]) if pid in enc_map and enc_map[pid] else np.nan
                    for pid in patient_col
                ]
            else:
                df[col_name] = patient_col

        elif col_type == 'boolean':
            p_true = props.get('probability', 0.5)
            df[col_name] = np.random.choice([0, 1], size=total_rows, p=[1 - p_true, p_true])

        elif col_type == 'unique_id':
            df[col_name] = np.random.randint(1000000, 9999999, size=total_rows)

        elif col_type == 'int':
            df[col_name] = np.random.randint(props['range'][0], props['range'][1] + 1, size=total_rows)

        elif col_type == 'float':
            df[col_name] = np.round(np.random.uniform(props['range'][0], props['range'][1], size=total_rows), 2)

        elif col_type == 'enumerated':
            df[col_name] = np.random.choice(props['values'], size=total_rows)

        elif col_type == 'date':
            start_date = pd.to_datetime(props['start'])
            end_date = pd.to_datetime(props['end'])
            days_diff = max(1, (end_date - start_date).days)
            random_days = np.random.randint(0, days_diff, size=total_rows)
            # FIX: Use .date on the resulting DatetimeIndex
            df[col_name] = (start_date + pd.to_timedelta(random_days, unit='d')).date

        elif col_type == 'time':
            hours = np.random.randint(0, 24, size=total_rows)
            minutes = np.random.randint(0, 60, size=total_rows)
            df[col_name] = [f"{h:02d}:{m:02d}:00" for h, m in zip(hours, minutes)]

        elif col_type == 'date_offset':
            base_dates = pd.to_datetime(df[props['base_col']])
            offsets = np.random.randint(props['days_range'][0], props['days_range'][1] + 1, size=total_rows)
            # FIX: Series requires .dt.date
            df[col_name] = (base_dates + pd.to_timedelta(offsets, unit='d')).dt.date

        elif col_type == 'categorical_tuple':
            choices = props['values']
            selected = [random.choice(choices) for _ in range(total_rows)]
            for i, target_col in enumerate(props['columns']):
                df[target_col] = [item[i] for item in selected]

    return df


def generate_eav_timeseriesv2(unique_ids, custom_tables, table_config, clinical_concepts, freq, days, primary_key_col):
    """Generates a long-format Entity-Attribute-Value (EAV) timeseries table dynamically."""
    schema = table_config.get('schema', {})
    source_path = table_config.get('source', '').split('.')
    concepts = clinical_concepts
    for key in source_path:
        concepts = concepts.get(key, {})

    periods = days * (24 // int(freq[:-1] if freq[:-1].isdigit() else 1))
    timestamps = pd.date_range('2024-01-01', periods=periods, freq=freq)

    idx = pd.MultiIndex.from_product(
        [unique_ids, timestamps, list(concepts.keys())],
        names=[primary_key_col, 'timestamp', 'test_key']
    )
    df = pd.DataFrame(index=idx).reset_index()

    # FIX: Efficiently generate values without using .apply(axis=1) which was crashing
    val_map = {k: v.get('range', [0, 1]) for k, v in concepts.items()}
    df['value'] = [
        np.round(np.random.uniform(val_map[k][0], val_map[k][1]), 2)
        for k in df['test_key']
    ]

    probs = df['test_key'].map(lambda k: concepts[k].get('prob', 0.5))
    df = df[np.random.rand(len(df)) < probs].copy()

    expected_metadata = ['normal_low', 'normal_high', 'order_code', 'order_name', 'department']
    for test_key, meta in concepts.items():
        missing = [m for m in expected_metadata if m not in meta]
        if missing:
            print(
                f"  ⚠️ Warning: Clinical concept '{test_key}' is missing metadata: {', '.join(missing)}. Safe defaults applied.")

    df['concept.code'] = df['test_key'].map(lambda k: concepts[k].get('code', 'UNKNOWN_CODE'))
    df['concept.name'] = df['test_key'].map(lambda k: concepts[k].get('name', 'Unknown Test'))
    df['concept.unit'] = df['test_key'].map(lambda k: concepts[k].get('unit', ''))
    df['concept.normal_low'] = df['test_key'].map(lambda k: concepts[k].get('normal_low', np.nan))
    df['concept.normal_high'] = df['test_key'].map(lambda k: concepts[k].get('normal_high', np.nan))
    df['concept.order_code'] = df['test_key'].map(lambda k: concepts[k].get('order_code', 'UNKNOWN_ORDER'))
    df['concept.order_name'] = df['test_key'].map(lambda k: concepts[k].get('order_name', 'Unspecified Order'))
    df['concept.department'] = df['test_key'].map(lambda k: concepts[k].get('department', 'Unspecified Dept'))

    df_final = pd.DataFrame()
    for col_name, rules in schema.items():
        map_target = rules.get('map_to')

        if map_target == primary_key_col:
            df_final[col_name] = df[primary_key_col]
        elif map_target == 'foreign_key':
            source_table = rules.get('source_table')
            if source_table and source_table in custom_tables:
                episodes = custom_tables[source_table]
                fk_col = [c for c in episodes.columns if 'ID' in c or 'IDENTIFIER' in c][0]
                pat_col = [c for c in episodes.columns if episodes[c].isin(unique_ids).any()][0]
                enc_map = episodes.groupby(pat_col)[fk_col].apply(list).to_dict()
                df_final[col_name] = [
                    np.random.choice(enc_map[pid]) if pid in enc_map and enc_map[pid] else np.nan
                    for pid in df[primary_key_col]
                ]
        elif map_target == 'unique_id':
            df_final[col_name] = np.random.randint(10000000, 99999999, size=len(df))
        elif map_target == 'timestamp':
            df_final[col_name] = df['timestamp']
        elif map_target == 'timestamp_offset':
            base_col = rules.get('base_col')
            hours_range = rules.get('hours_range', [1, 24])
            offsets = pd.to_timedelta(np.random.randint(hours_range[0], hours_range[1] + 1, size=len(df)), unit='h')
            df_final[col_name] = df_final[base_col] + offsets
        elif map_target == 'value':
            df_final[col_name] = df['value']
        elif map_target and map_target.startswith('concept.'):
            df_final[col_name] = df[map_target]

    if not df_final.empty:
        first_col = list(schema.keys())[0]
        time_cols = [k for k, v in schema.items() if v.get('map_to') == 'timestamp']
        sort_cols = [first_col] + time_cols
        df_final = df_final.sort_values(sort_cols).reset_index(drop=True)

    return df_final

if __name__ == '__main__':
    # --------------------------
    # 1. Load Configuration
    # --------------------------
    # Assuming you run this from inside the src/ folder
    config_path = Path('../../config/data_config_old.yaml')

    # Fallback/mock config if file doesn't exist yet for testing
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Please ensure your YAML file is created.")
        exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    params = config.get('generation_params', {})
    n_patients = params.get('n_patients', 100)
    days = params.get('days', 10)
    freq = params.get('freq', '4h')
    output_format = params.get('output_format', 'tidy')

    # --------------------------
    # 2. Generate data
    # --------------------------
    print(f"Generating data for {n_patients} patients...")
    df_static, df_ts = generate_clinical_data(
        config=config,
        n_patients=n_patients,
        days=days,
        freq=freq,
        output_format=output_format
    )

    print('\n--- Sample: Static Demographics ---')
    print(df_static.head())
    print('\n--- Sample: Timeseries Data ---')
    print(df_ts.head())

    # --------------------------
    # 3. Add missingness
    # --------------------------
    print("\nApplying missingness masks...")
    df_ms = apply_missingness(df_ts, ts_config=config.get('ts_config', {}))

    # --------------------------
    # 4. Save data in structured FS
    # --------------------------
    date_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = Path(f'../data/synthetic/{date_str}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving files to <{output_dir}>...")
    df_static.to_csv(output_dir / 'df_static.csv', index=False)
    df_ts.to_csv(output_dir / 'df_ts.csv', index=True)
    df_ms.to_csv(output_dir / 'df_ts_missing.csv', index=True)
    print("Done!")

