"""
cinnabar
Report results for free energy simualtions
"""


# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

from . import stats
from . import wrangle
from .wrangle import (
    RelativeResult,
    ExperimentalResult,
    FEMap, read_csv,
)
from . import plotlying
from . import plotting

# the cli entrypoint
from . import arsenic
