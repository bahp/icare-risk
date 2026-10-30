# Libraries
import yaml
import shutil
import importlib.resources

from pathlib import Path
from typing import Any, Dict

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


def _copy_pkg_files(
        source_subpath: str,
        target_dir: str | Path,
        extension: str = "",
        overwrite: bool = False,
        verbose: bool = True
) -> int:
    """
    Copy a subset of files from a package subdirectory to a target directory.
    If extension is empty (""), copies all files recursively.

    Parameters
    ----------
    source_subpath : str
        Relative path inside the package root pointing to the source directory.
    target_dir : str or Path
        Destination folder where matching files will be copied.
    extension : str, default=""
        File extension filter (e.g., '.yaml', '.ipynb'). If empty, matches all files.
    overwrite : bool, default=False
        Whether to overwrite existing files at the destination.
    verbose : bool, default=True
        Whether to print logging messages containing copied file names and counts.

    Returns
    -------
    int
        Number of files successfully copied/overwritten.
    """
    target, source = Path(target_dir), get_pkg_path(source_subpath)

    if not source.is_dir():
        if verbose:
            print(f"Error: Source directory not found at {source}")
        return 0

    copied_files = []
    pattern = f"*{extension}" if extension else "*"

    # Use rglob to recursively find files matching the extension across subfolders
    for item in source.rglob(pattern):
        if item.is_file():
            # Calculate relative path to preserve directory structure
            rel_path = item.relative_to(source)
            dest = target / rel_path

            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or overwrite:
                shutil.copy2(item, dest)
                copied_files.append(str(rel_path))

    if verbose:
        print(f"Successfully copied/updated {len(copied_files)} file(s) to {target.resolve()}:")
        for name in copied_files:
            print(f"  - {name}")

    return len(copied_files)

def extract_notebooks(target_dir: str | Path = ".",
                      extension: str = ".ipynb",
                      overwrite: bool = False,
                      verbose: bool = True) -> int:
    """
    Copy bundled pipeline notebooks from the installed package to the target directory.
    """
    return _copy_pkg_files("notebooks",
        target_dir=target_dir, extension=extension,
        overwrite=overwrite, verbose=verbose)


def extract_configs(target_dir: str | Path = "config/icare",
                    extension: str = ".yaml",
                    overwrite: bool = False,
                    verbose: bool = True) -> int:
    """
    Copy bundled YAML configuration files from the installed package to the target directory.
    """
    return _copy_pkg_files("config/icare",
        target_dir=target_dir, extension=extension,
        overwrite=overwrite, verbose=verbose)