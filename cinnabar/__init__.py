"""
cinnabar
Report results for free energy simulations
"""

from importlib.metadata import version

__version__ = version("cinnabar")

from cinnabar import stats
from cinnabar.femap import FEMap, unit
from cinnabar.measurements import Measurement, ReferenceState
# from cinnabar. import plotting
