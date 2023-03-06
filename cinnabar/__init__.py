"""
cinnabar
Report results for free energy simulations
"""


# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

from .measurements import AbsoluteMeasurement, RelativeMeasurement
from .femap import FEMap, unit
from . import stats
# from . import plotting