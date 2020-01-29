"""
FreeEnergyFramework
Report results for free energy simualtions
"""

# Add imports here
from .read import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
