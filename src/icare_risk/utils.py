import sys
import yaml
import pandas as pd
from pathlib import Path
from importlib.resources import files


def deep_merge(base_dict, override_dict, path=None, strict=False):
    """
    Recursively deep-merges override_dict into base_dict.
    Alerts the user if a key from override_dict overwrites an
    existing key in base_dict.

    Parameters
    ----------
    base_dict : dict
        The default package configuration.
    override_dict : dict
        The user's local override configuration.
    path : list, optional
        Used internally to track the key hierarchy for logging.
    strict : bool, default=False
        If True, raises a ValueError on duplicate keys instead of just warning.
    """
    if path is None:
        path = []

    for key, value in override_dict.items():
        current_path = path + [str(key)]

        # If the key exists in both dictionaries
        if key in base_dict:
            # If both are dictionaries, recurse deeper
            if isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value, current_path, strict)

            # If they are not both dictionaries, an overwrite is happening
            else:
                key_string = " -> ".join(current_path)
                msg = f"⚠️ Override Alert: The key '{key_string}' in your local config is overwriting the default iCARE setting."

                if strict:
                    raise ValueError(f"❌ Strict Mode Error: {msg}")
                else:
                    print(msg)

                # Perform the overwrite
                base_dict[key] = value

        # If the key is entirely new, simply add it
        else:
            base_dict[key] = value

    return base_dict


def load_yaml_config(config_name="data_config.yaml",
                     user_path=None,
                     strict_merge=False):
    """
    Loads a default YAML from the package and deep-merges user overrides.
    """
    # 1. Read the default YAML from inside the installed package
    default_yaml = files('icare_risk.config').joinpath(config_name)
    print(f"⚙️ Loaded default config from: {default_yaml}")


    with default_yaml.open('r') as f:
        config = yaml.safe_load(f)

    # 2. If the user provided an override path, load it and merge
    if user_path:
        user_file = Path(user_path)
        if user_file.is_file():
            print(f"⚙️ Overriding default config using: {user_file.resolve()}")
            with user_file.open('r') as f:
                user_overrides = yaml.safe_load(f) or {}

            config = deep_merge(config, user_overrides, strict=strict_merge)
        else:
            print(f"⚠️ Warning: Override file '{user_path}' not found. Using defaults.")
    else:
        print(f"⚙️ No override config provided. Running solely on defaults.")

    return config

#def load_data_config():
#    """Loads data_config.yaml dynamically relative to CWD or project root."""
#    config_path = Path.cwd() / 'config' / 'data_config.yaml'
#    if not config_path.exists():
#        config_path = project_root / 'config' / 'data_config.yaml'##
#
#    if not config_path.exists():
#        raise FileNotFoundError(f"Configuration file not found at {config_path}")#
#
#    with open(config_path, 'r') as file:
#        return yaml.safe_load(file)


def get_latest_data_dir(user_config_path=None,
                        loaded_config=None):
    """
    Finds the most recently created raw data directory inside data/synthetic/
    or data/external/ depending on the 'data_source' key in data_config.yaml.
    """
    if loaded_config is not None:
        config = loaded_config
    else:
        config = load_yaml_config("data_config.yaml", user_config_path)

    source = config.get('data_source', 'synthetic').lower()

    # Determine base directory based on config
    paths = config.get('paths', {})
    if source == 'external':
        rel_path = paths.get('external_dir', 'data/external')
    else:
        rel_path = paths.get('synthetic_dir', 'data/synthetic')

    synthetic_dir = Path.cwd() / rel_path

    if not synthetic_dir.exists():
        raise FileNotFoundError(
            f"No [{source}] data root folder found at '{synthetic_dir}'. "
            f"Please ensure '{rel_path}' exists."
        )

    dirs = sorted([d for d in synthetic_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(
            f"No timestamped directories found inside [{source}] folder: {synthetic_dir}"
        )

    latest_dir = dirs[-1]
    print(f"🔄 Active Data Source: [{source.upper()}] -> Reading from '{latest_dir.name}'")
    return latest_dir


def get_latest_processed_file(user_config_path=None):
    """Finds the most recently processed dataset (any .csv file in the latest processed folder)."""
    config = load_yaml_config("data_config.yaml", user_config_path)
    paths = config.get('paths', {})

    # Resolve processed directory using config or default fallback
    rel_processed_path = paths.get('processed_dir', 'data/processed')
    processed_dir = Path.cwd() / rel_processed_path

    if not processed_dir.exists():
        raise FileNotFoundError(f"No processed data found at {processed_dir}. Run 02_build_features_icare.py first.")

    # Check if files exist directly or inside subdirectories
    csv_files = list(processed_dir.glob('*.csv'))

    if csv_files:
        # Flat structure: grab the latest modified CSV file in processed_dir
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"📄 Loading processed dataset: '{latest_file.name}' from '{processed_dir.name}'")
        return latest_file

    # Nested structure: look inside timestamp subdirectories (e.g., data/processed/YYYY-MM-DD_HHMMSS/)
    dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No processed data directories or CSV files found in {processed_dir}.")

    latest_dir = dirs[-1]
    csv_files_nested = list(latest_dir.glob('*.csv'))

    if not csv_files_nested:
        raise FileNotFoundError(f"No CSV files found in the directory: {latest_dir}")

    latest_file = csv_files_nested[0]
    print(f"📄 Loading processed dataset: '{latest_file.name}' from run '{latest_dir.name}'")

    return latest_file



def validate_required_columns(df,
                              required_cols,
                              score_name="Clinical Score",
                              strict=False):
    """
    Validates the presence of required clinical variables in a DataFrame.

    This utility prevents the 'silent zero' problem, where a missing column
    leads to an incorrectly low clinical score (under-scoring). It logs
    missing columns to the console to ensure data integrity during
    feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataset being evaluated.
    required_cols : list of str
        The column names mandatory for the specific scoring system.
    score_name : str, default="Clinical Score"
        The name of the score (e.g., 'INCREMENT-ESBL') for log identification.
    strict : bool, default=False
        If True, raises a ValueError when columns are missing.
        If False, prints a warning and allows the pipeline to continue.

    Returns
    -------
    bool
        True if all required columns are present; False if any are missing.

    Raises
    ------
    ValueError
        If 'strict' is True and required columns are missing from the input DataFrame.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        error_msg = f"[{score_name}] Missing critical variables: {missing_cols}"
        if strict:
            raise ValueError(f"❌ {error_msg}. Pipeline halted to prevent data corruption.")
        print(f"⚠️ Warning: {error_msg}. Results will be underestimated for these records.")
        return False

    return True


def check_col_bool(df, col_name):
    """Returns a boolean Series checking if a binary column is 1/True."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col_name].fillna(0).astype(int) == 1


def check_col_contains(df, col_name, keywords):
    """Returns a boolean Series checking if text in a column contains certain keywords."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    pattern = '|'.join(keywords) # Create a regex pattern: 'keyword1|keyword2'
    return df[col_name].astype(str).str.lower().str.contains(pattern, na=False)


def check_col_threshold(df, col_name, threshold, operator='>'):
    """Returns a boolean Series checking if a numerical column meets a threshold."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    # Temporarily fill NaNs with a safe value that won't trigger the threshold
    temp_col = df[col_name].fillna(-9999 if operator == '>' else 9999)

    if operator == '>': return temp_col > threshold
    if operator == '>=': return temp_col >= threshold
    if operator == '<': return temp_col < threshold
    if operator == '<=': return temp_col <= threshold
    return pd.Series(False, index=df.index)


def check_col_icd10(df, col_name, target_codes):
    """Returns a boolean Series checking if any ICD codes match the target list."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    def match_codes(patient_codes):
        if pd.isna(patient_codes): return False
        patient_codes = [c.strip() for c in str(patient_codes).split(',')]
        return any(any(str(pc).startswith(str(tc)) for tc in target_codes) for pc in patient_codes)

    return df[col_name].apply(match_codes)