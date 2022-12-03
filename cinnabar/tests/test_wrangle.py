from importlib import resources
import pytest

from cinnabar import wrangle


@pytest.fixture
def example_csv():
    with resources.path('cinnabar.data', 'example.csv') as f:
        yield f


def test_read_csv(example_csv):
    data = wrangle.read_csv(example_csv)

    assert 'Experimental' in data
    assert len(data['Experimental']) == 36

    assert 'Calculated' in data
    assert len(data['Calculated']) == 58
