import os
import sys
from pathlib import Path
from typing import Union

from icare_risk.utils.io import (
    get_project_root,
    extract_configs, 
    extract_notebooks, 
    _copy_pkg_files
)


# -------------------------------
# Root & Directories
# -------------------------------
# Automatically inject the root into sys.path the second anyone imports icare_risk!
#PROJECT_ROOT = get_project_root(strict=False)
#if str(PROJECT_ROOT) not in sys.path:
#    sys.path.append(str(ROOT))

PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROC_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Centralized Config
DB_CFG = CONFIG_DIR / "generate_db.yaml"
CODE_SEARCH_CFG = CONFIG_DIR /"code_search.yaml"
SCHEMA_YAML_PATH = CONFIG_DIR / "schema.yaml"
ANALYSIS_YAML_PATH = CONFIG_DIR / "analysis.yaml"
PHENOTYPES_YAML_PATH = CONFIG_DIR / "phenotypes.yaml"

# Lookups
LOOKUP_3GCR_PATH = CONFIG_DIR / "lookups" / "3GCR.csv"
LOOKUP_3GCS_PATH = CONFIG_DIR / "lookups" / "3GCS.csv"

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "10000p"

def get_path(relative_path: str) -> Path:
    """Utility to securely join any path relative to the project root."""
    return (PROJECT_ROOT / relative_path.lstrip("/\\")).resolve()

def get_data_root_from_schema(
    schema_path: Union[str, Path] = SCHEMA_YAML_PATH,
    default_dir: Union[str, Path] = DEFAULT_DATA_DIR,
) -> Path:
    """Retrieves the data_root from the schema YAML configuration file,

    falling back to the default directory if the file or key is missing.
    """
    #
    from icare_risk.utils.io import load_yaml_file
    import yaml

    yaml_path = Path(schema_path)

    # If relative, resolve it relative to the project root
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                if config.get("data_root"):
                    resolved_root = Path(config["data_root"])
                    # Handle relative paths inside the YAML as relative to project root
                    return (
                        resolved_root
                        if resolved_root.is_absolute()
                        else PROJECT_ROOT / resolved_root
                    )
        except Exception as e:
            print(f"⚠️ Error reading schema config at {yaml_path}: {e}")

    print(f"⚠️ Using default data root: {default_dir}")
    return Path(default_dir)


def print_paths_summary(raise_on_missing: bool = False) -> dict[str, Path]:
    """
    Prints a clear, audited summary of all central project paths and their 
    current existence status on disk. 

    Parameters
    ----------
    raise_on_missing : bool, default=False
        If True, raises a FileNotFoundError if any critical file/directory is missing.

    Returns
    -------
    dict[str, Path]
        A dictionary mapping path identifiers to their resolved Path objects.
    """
    paths_map = {
        "Project Root": PROJECT_ROOT,
        "Config Directory": CONFIG_DIR,
        "Data Directory": DATA_DIR,
        #"Processed Data Dir": DATA_PROC_DIR,
        "Output Directory": OUTPUT_DIR,
        "Database Config": DB_CFG,
        "Code Search Config": CODE_SEARCH_CFG,
        "Schema YAML": SCHEMA_YAML_PATH,
        "Analysis YAML": ANALYSIS_YAML_PATH,
        "Phenotypes YAML": PHENOTYPES_YAML_PATH,
        "Lookup 3GCR": LOOKUP_3GCR_PATH,
        "Lookup 3GCS": LOOKUP_3GCS_PATH,
    }

    missing_count = 0
    print("\n" + "=" * 70)
    print(f"{' ICARE_RISK WORKSPACE PATHS SUMMARY ':=^70}")
    print("=" * 70)
    
    for name, path_obj in paths_map.items():
        exists = path_obj.exists()
        if not exists and name != "Output Directory":  # Output dir might be created dynamically
            missing_count += 1
            
        status = "[✓] Found   " if exists else "[✗] MISSING "
        kind = "📁 DIR " if path_obj.suffix == "" else "📄 FILE"
        
        print(f"  {status} {kind} | {name:<22}: {path_obj.resolve()}")
        
    print("=" * 70)
    
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} required path(s) were not found on disk.")
        if raise_on_missing:
            raise FileNotFoundError(f"[Paths Error] {missing_count} required configuration or data paths are missing.")
    else:
        print("✨ All workspace paths successfully verified and ready.")
    print("=" * 70 + "\n")
    
    return paths_map

def print_directory_content(dir):
    print(f"Target directory: {dir}")
    print(f"Directory exists? {dir.exists()}")

    if dir.exists():
        print(f"\n--- Files found in '{dir}' ---")
        for item in dir.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    else:
        print("\n⚠️ The directory path does not exist on this machine/environment!")

    return dir
