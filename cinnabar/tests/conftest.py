import pytest
from importlib import resources

from cinnabar import FEMap


@pytest.fixture(scope='session')
def example_csv():
    with resources.path("cinnabar.data", "example.csv") as fn:
        yield str(fn)


@pytest.fixture(scope="session")
def fe_map(example_csv):
    """FEMap using test csv data"""
    return FEMap.from_csv(example_csv)
