"""
cinnabar
Report results for free energy simulations
"""
from importlib.metadata import version
__version__ = version("cinnabar")

from .measurements import ReferenceState, Measurement
from .femap import FEMap, unit
from . import stats
# from . import plotting