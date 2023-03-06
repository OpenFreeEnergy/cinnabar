from importlib import resources
import pytest

from openff.units import unit
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


def test_femap():
    m = wrangle.FEMap()

    m1 = wrangle.RelativeMeasurement(labelA='ligA', labelB='ligB', DDG=1.1 * unit.kilojoule_per_mole,
                                     uncertainty=0.1 * unit.kilojoule_per_mole, computational=True)
    m2 = wrangle.AbsoluteMeasurement(label='ligA', DG=10.0 * unit.kilojoule_per_mole,
                                     uncertainty=0.2 * unit.kilojoule_per_mole, computational=False)
    m3 = wrangle.AbsoluteMeasurement(label='ligB', DG=11.0 * unit.kilojoule_per_mole,
                                     uncertainty=0.3 * unit.kilojoule_per_mole, computational=False)

    m.add_measurement(m1)
    m.add_measurement(m2)
    m.add_measurement(m3)

    assert m.n_ligands == 2
