import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


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

    # FIX: Correct Path Resolution for clinical_concepts
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
    val_map = {k: v.get('range', [0, 1]) for k, v in concepts.items()}
    df['value'] = [np.round(np.random.uniform(val_map[k][0], val_map[k][1]), 2) for k in df['test_key']]

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