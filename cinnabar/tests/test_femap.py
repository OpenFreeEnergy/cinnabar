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
    assert len(example_map.graph.edges) == (58 + 36) * 2


def test_degree(example_map):
    assert example_map.degree == pytest.approx(58/36)


def test_weakly_connected(example_map):
    assert example_map.check_weakly_connected() is True


def test_femap():
    m = cinnabar.FEMap()

    m1 = cinnabar.Measurement(labelA='ligA', labelB='ligB', DG=1.1 * unit.kilojoule_per_mole,
                              uncertainty=0.1 * unit.kilojoule_per_mole, computational=True)

    g = cinnabar.ReferenceState()
    m2 = cinnabar.Measurement(labelA=g, labelB='ligA', DG=10.0 * unit.kilojoule_per_mole,
                              uncertainty=0.2 * unit.kilojoule_per_mole, computational=False)
    m3 = cinnabar.Measurement(labelA=g, labelB='ligB', DG=11.0 * unit.kilojoule_per_mole,
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

    edges = list(example_map.graph[cinnabar.ReferenceState()])
    for e in edges:
        data = example_map.graph.get_edge_data(cinnabar.ReferenceState(label='MLE'), e)
        # grab the dict containing MLE data
        for _, d in data.items():
            if d['source'] == 'MLE':
                break

        y = d['DG']
        yerr = d['uncertainty']

        y_ref, yerr_ref = ref_mle_results[e]

        assert y.magnitude == pytest.approx(y_ref), e
        assert yerr.magnitude == pytest.approx(yerr_ref), e
