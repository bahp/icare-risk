from pathlib import Path
import importlib.resources
from typing import Any, Dict
import yaml


def get_pkg_path(
    relative_path: str = "",
    package_name: str = "icare_risk"
) -> Path:
    """
    Resolve the absolute filesystem path for a package resource or asset.

    Parameters
    ----------
    relative_path : str, default=""
        Path relative to the root module directory (e.g., "config/schema.yaml").
    package_name : str, default="icare_risk"
        Name of the target root package module.

    Returns
    -------
    Path
        Absolute `pathlib.Path` pointing to the target file or directory.

    Raises
    ------
    ModuleNotFoundError
        If `package_name` is not an installed or importable Python module.

    Examples
    --------
    >>> get_pkg_path("config/schema.yaml")
    PosixPath('/app/src/icare_risk/config/schema.yaml')

    >>> get_pkg_path()
    PosixPath('/app/src/icare_risk')
    """
    base_path = importlib.resources.files(package_name)
    return Path(str(base_path / relative_path))


def load_pkg_yaml(
    relative_path: str,
    package_name: str = "icare_risk"
) -> Dict[str, Any]:
    """
    Load and parse a YAML file from package resources.

    Parameters
    ----------
    relative_path : str
        Path relative to the package root pointing to a YAML file.
    package_name : str, default="icare_risk"
        Name of the target root package module.

    Returns
    -------
    dict
        Parsed YAML content as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    yaml.YAMLError
        If the target file contains invalid YAML syntax.

    Examples
    --------
    >>> schema_dict = load_pkg_yaml("config/schema.yaml")
    >>> "episodes" in schema_dict
    True
    """
    yaml_path = get_pkg_path(relative_path, package_name=package_name)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)