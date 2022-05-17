import pytest
from importlib import resources

from arsenic.wrangle import FEMap


@pytest.fixture(scope="session")
def fe_map():
    """FEMap using test csv data"""

    with resources.path("arsenic.data", "example.csv") as fn:
        femap = FEMap(fn)

    return femap
