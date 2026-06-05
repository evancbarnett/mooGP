from importlib.metadata import version, PackageNotFoundError

from .test import test

try:
    __version__ = version("moogp")
except PackageNotFoundError:
    # package is not installed
    pass
