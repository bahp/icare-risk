from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("icare-risk")
except PackageNotFoundError:
    __version__ = "unknown"