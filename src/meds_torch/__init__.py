from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meds-torch")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
