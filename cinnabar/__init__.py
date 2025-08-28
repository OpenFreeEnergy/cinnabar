"""
cinnabar
Report results for free energy simulations
"""

from importlib.metadata import version

__version__ = version("cinnabar")

from . import stats
from .femap import FEMap, unit
from .measurements import Measurement, ReferenceState
# from . import plotting
