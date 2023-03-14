from importlib import resources
import pytest
import json
import networkx as nx
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
    assert example_map.n_edges == 58
    assert len(example_map.graph.edges) == 58 + 36


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


def test_to_legacy(example_map, ref_legacy):
    # checks contents of legacy graph output against reference
    g = example_map.to_legacy_graph()

    s = json.dumps(nx.to_dict_of_dicts(g),
                   indent=1, sort_keys=True,
                   default=lambda x: x.magnitude)  # removes units

    assert s == ref_legacy


def test_generate_absolute_values(example_map, ref_mle_results):
    example_map.generate_absolute_values()

    edges = list(example_map.graph['NULL'])
    for e in edges:
        data = example_map.graph.get_edge_data('NULL', e)
        # grab the dict containing MLE data
        for _, d in data.items():
            if d['source'] == 'MLE':
                break

        y = d['DDG']
        yerr = d['uncertainty']

        y_ref, yerr_ref = ref_mle_results[e]

        assert y.magnitude == pytest.approx(y_ref), e
        assert yerr.magnitude == pytest.approx(yerr_ref), e


@pytest.fixture(scope='session')
def everything_map():
    m = cinnabar.FEMap()

    m.add_measurement(
        cinnabar.RelativeMeasurement(
            labelA='alpha', labelB='beta',
            DDG=11.0 * unit.kilocalorie_per_mole, uncertainty=1.2 * unit.kilocalorie_per_mole,
            computational=True, source='theft',
        )
    )
    m.add_measurement(
        cinnabar.RelativeMeasurement(
            labelA='alpha', labelB='gamma',
            DDG=12.0 * unit.kilocalorie_per_mole, uncertainty=1.4 * unit.kilocalorie_per_mole,
            computational=True, source='found',
        )
    )
    m.add_measurement(
        cinnabar.AbsoluteMeasurement(
            label='beta',
            DG=24.0 * unit.kilocalorie_per_mole, uncertainty=1.6 * unit.kilocalorie_per_mole,
            computational=True, source='found',
        )
    )
    m.add_measurement(
        cinnabar.AbsoluteMeasurement(
            label='gamma',
            DG=24.0 * unit.kilocalorie_per_mole, uncertainty=1.6 * unit.kilocalorie_per_mole,
            computational=False, source='literature',
        )
    )

    return m


class TestFEMapGet:
    def test_absolute_comp(self, everything_map):
        res = list(everything_map.get_absolute_measurements())
        expt_res = list(everything_map.get_absolute_measurements(computational=False))
        comp_res = list(everything_map.get_absolute_measurements(computational=True))

        assert len(res) == 2
        assert len(expt_res) == 1
        assert len(comp_res) == 1

    def test_absolute_source(self, everything_map):
        res = list(everything_map.get_absolute_measurements(source='literature'))

        assert len(res) == 1
        assert res[0].label == 'gamma'

    def test_absolute_label(self, everything_map):
        res1 = list(everything_map.get_absolute_measurements(label='delta'))
        res2 = list(everything_map.get_absolute_measurements(label='beta'))

        assert len(res1) == 0
        assert len(res2) == 1
        assert res2[0].source == 'found'

    def test_relative_comp(self, everything_map):
        res = list(everything_map.get_relative_measurements())
        expt_res = list(everything_map.get_relative_measurements(computational=False))
        comp_res = list(everything_map.get_relative_measurements(computational=True))

        assert len(res) == 2
        assert len(expt_res) == 0
        assert len(comp_res) == 2

    def test_relative_source(self, everything_map):
        res = list(everything_map.get_relative_measurements(source='theft'))

        assert len(res) == 1
        assert res[0].labelA == 'alpha'

    def test_relative_label(self, everything_map):
        res1 = list(everything_map.get_relative_measurements(label='delta'))
        res2 = list(everything_map.get_relative_measurements(label='alpha'))

        assert len(res1) == 0
        assert len(res2) == 2
        assert {m.labelB for m in res2} == {'beta', 'gamma'}
