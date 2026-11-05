from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("icare-risk")
except PackageNotFoundError:
    __version__ = "unknown"

from icare_risk.utils.io import (
    setup_workspace, 
    extract_configs, 
    extract_notebooks
)