import pytest
from importlib import resources

from cinnabar.wrangle import FEMap


@pytest.fixture(scope="session")
def fe_map():
    """FEMap using test csv data"""

    with resources.path("cinnabar.data", "example.csv") as fn:
        femap = FEMap(fn)

    return femap
