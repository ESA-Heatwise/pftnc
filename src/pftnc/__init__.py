from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

try:
    __version__ = package_version("pftnc")
except PackageNotFoundError:
    # Keep source checkouts importable when the project has not been installed.
    __version__ = "unknown"
