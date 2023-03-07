from importlib import resources
import pytest

from openff.units import unit

import cinnabar
from cinnabar import femap


def test_read_csv(example_csv):
    data = femap.read_csv(example_csv)

    assert 'Experimental' in data
    assert len(data['Experimental']) == 36

    assert 'Calculated' in data
    assert len(data['Calculated']) == 58


@pytest.fixture(scope='session')
def example_map(example_csv):
    return cinnabar.FEMap.from_csv(example_csv)


def test_from_csv(example_map):
    assert example_map.n_ligands == 36
    assert len(example_map.computational_graph.edges) == 58
    assert len(example_map.experimental_graph.edges) == 36


def test_degree(example_map):
    assert example_map.degree == pytest.approx(58/36)


def test_weakly_connected(example_map):
    assert example_map.check_weakly_connected() is True


def test_femap():
    m = cinnabar.FEMap()

    m1 = cinnabar.RelativeMeasurement(labelA='ligA', labelB='ligB', DDG=1.1 * unit.kilojoule_per_mole,
                                      uncertainty=0.1 * unit.kilojoule_per_mole, computational=True)
    m2 = cinnabar.AbsoluteMeasurement(label='ligA', DG=10.0 * unit.kilojoule_per_mole,
                                      uncertainty=0.2 * unit.kilojoule_per_mole, computational=False)
    m3 = cinnabar.AbsoluteMeasurement(label='ligB', DG=11.0 * unit.kilojoule_per_mole,
                                      uncertainty=0.3 * unit.kilojoule_per_mole, computational=False)

    m.add_measurement(m1)
    m.add_measurement(m2)
    m.add_measurement(m3)

    assert m.n_ligands == 2
