import json
from importlib import resources

import networkx as nx
import pytest
from openff.units import unit

import cinnabar
from cinnabar import femap


def test_read_csv(example_csv):
    data = femap.read_csv(example_csv)

    assert "Experimental" in data
    assert len(data["Experimental"]) == 36

    assert "Calculated" in data
    assert len(data["Calculated"]) == 58


@pytest.fixture()
def example_map(example_csv):
    return cinnabar.FEMap.from_csv(example_csv)


def test_from_csv(example_map):
    assert example_map.n_ligands == 36
    assert example_map.n_edges == 58
    assert len(example_map._graph.edges) == (58 + 36) * 2


def test_eq(example_csv):
    m1 = cinnabar.FEMap.from_csv(example_csv)
    m2 = cinnabar.FEMap.from_csv(example_csv)
    m3 = cinnabar.FEMap.from_csv(example_csv)
    m3.add_experimental_measurement(
        label="this",
        value=4.2 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )

    assert m1 == m2
    assert m1 != m3


def test_degree(example_map):
    assert example_map.degree == pytest.approx(58 / 36)


def test_weakly_connected(example_map):
    assert example_map.check_weakly_connected() is True


def test_femap_add_measurement():
    m = cinnabar.FEMap()

    m1 = cinnabar.Measurement(
        labelA="ligA",
        labelB="ligB",
        DG=1.1 * unit.kilojoule_per_mole,
        uncertainty=0.1 * unit.kilojoule_per_mole,
        computational=True,
    )

    g = cinnabar.ReferenceState()
    m2 = cinnabar.Measurement(
        labelA=g,
        labelB="ligA",
        DG=10.0 * unit.kilojoule_per_mole,
        uncertainty=0.2 * unit.kilojoule_per_mole,
        computational=False,
    )
    m3 = cinnabar.Measurement(
        labelA=g,
        labelB="ligB",
        DG=11.0 * unit.kilojoule_per_mole,
        uncertainty=0.3 * unit.kilojoule_per_mole,
        computational=False,
    )

    m.add_measurement(m1)
    m.add_measurement(m2)
    m.add_measurement(m3)

    assert m.n_ligands == 2
    assert set(m.ligands) == {"ligA", "ligB"}


@pytest.mark.parametrize("ki", [False, True])
def test_femap_add_experimental(ki):
    ref_v = -9.58015754 * unit.kilocalorie_per_mole
    ref_u = 0.0594372794 * unit.kilocalorie_per_mole

    if ki:
        v = 100 * unit.nanomolar
        u = 10 * unit.nanomolar
    else:
        v = ref_v
        u = ref_u
    m = cinnabar.FEMap()

    m.add_experimental_measurement("ligA", v, u, source="voodoo", temperature=299.1 * unit.kelvin)

    assert set(m.ligands) == {"ligA"}
    d = m._graph.get_edge_data(cinnabar.ReferenceState(), "ligA")
    assert d.keys() == {0}
    d = d[0]
    assert d["computational"] is False
    assert d["source"] == "voodoo"
    assert d["temperature"] == 299.1 * unit.kelvin
    assert d["DG"].m == pytest.approx(ref_v.m)
    assert d["uncertainty"].m == pytest.approx(ref_u.m)


def test_femap_add_experimental_float_VE():
    v = -9.58015754
    u = 0.0594372794

    m = cinnabar.FEMap()

    with pytest.raises(ValueError, match="Must include units"):
        m.add_experimental_measurement("ligA", v, u, source="voodoo", temperature=299.1 * unit.kelvin)


@pytest.mark.parametrize("default_T", [True, False])
def test_add_ABFE(default_T):
    v = -9.58015754 * unit.kilocalorie_per_mole
    u = 0.0594372794 * unit.kilocalorie_per_mole
    T = 299.1 * unit.kelvin if not default_T else 298.15 * unit.kelvin
    m = cinnabar.FEMap()

    if default_T:
        m.add_absolute_calculation(label="foo", value=v, uncertainty=u, source="ebay")
    else:
        m.add_absolute_calculation(label="foo", value=v, uncertainty=u, source="ebay", temperature=T)

    assert set(m.ligands) == {"foo"}
    d = m._graph.get_edge_data(cinnabar.ReferenceState(), "foo")
    assert len(d) == 1
    d = d[0]
    assert d["DG"] == v
    assert d["uncertainty"] == u
    assert d["temperature"] == T


@pytest.mark.parametrize("default_T", [True, False])
def test_add_RBFE(default_T):
    v = -9.58015754 * unit.kilocalorie_per_mole
    u = 0.0594372794 * unit.kilocalorie_per_mole
    T = 299.1 * unit.kelvin if not default_T else 298.15 * unit.kelvin
    m = cinnabar.FEMap()

    if default_T:
        m.add_relative_calculation(labelA="foo", labelB="bar", value=v, uncertainty=u, source="ebay")
    else:
        m.add_relative_calculation(labelA="foo", labelB="bar", value=v, uncertainty=u, source="ebay", temperature=T)

    assert set(m.ligands) == {"foo", "bar"}
    d = m._graph.get_edge_data("foo", "bar")
    assert len(d) == 1
    d = d[0]
    assert d["DG"] == v
    assert d["uncertainty"] == u
    assert d["temperature"] == T


def test_to_legacy(example_map, ref_legacy):
    # checks contents of legacy graph output against reference
    g = example_map.to_legacy_graph()

    s = json.dumps(nx.to_dict_of_dicts(g), indent=1, sort_keys=True, default=lambda x: x.magnitude)  # removes units

    assert s == ref_legacy


def test_generate_absolute_values(example_map, ref_mle_results):
    example_map.generate_absolute_values()

    for e, (y_ref, yerr_ref) in ref_mle_results.items():
        data = example_map._graph.get_edge_data(cinnabar.ReferenceState(label="MLE"), e)
        # grab the dict containing MLE data
        for _, d in data.items():
            if d["source"] == "MLE":
                break

        y = d["DG"]
        yerr = d["uncertainty"]

        y_ref, yerr_ref = ref_mle_results[e]

        assert y.magnitude == pytest.approx(y_ref), e
        assert yerr.magnitude == pytest.approx(yerr_ref), e


def test_to_dataframe(example_map):
    abs_df = example_map.get_absolute_dataframe()
    rel_df = example_map.get_relative_dataframe()

    assert abs_df.shape == (36, 5)
    assert rel_df.shape == (58, 6)

    example_map.generate_absolute_values()

    abs_df2 = example_map.get_absolute_dataframe()
    assert abs_df2.shape == (72, 5)
    assert abs_df2.loc[abs_df2.computational].shape == (36, 5)
    assert abs_df2.loc[~abs_df2.computational].shape == (36, 5)


def test_to_networkx(example_map):
    g = example_map.to_networkx()

    assert g
    assert isinstance(g, nx.MultiDiGraph)
    # should have (exptl + comp edges) * 2
    assert len(g.edges) == 2 * (36 + 58)


def test_from_networkx(example_map):
    g = example_map.to_networkx()

    m2 = cinnabar.FEMap.from_networkx(g)

    assert example_map == m2


def test_add():
    m1 = cinnabar.FEMap()
    m1.add_experimental_measurement(
        label="c1",
        value=10.1 * unit.nanomolar,
        uncertainty=0.2 * unit.nanomolar,
    )
    m1.add_experimental_measurement(
        label="c2",
        value=10.2 * unit.nanomolar,
        uncertainty=0.3 * unit.nanomolar,
    )

    m2 = cinnabar.FEMap()
    m2.add_absolute_calculation(
        label="c1",
        value=-9.5 * unit.kilocalorie_per_mole,
        uncertainty=0.4 * unit.kilocalorie_per_mole,
    )

    m3 = m1 + m2

    assert len(m3) == 3
    measurements = set(m3)

    ref1 = set(m1)
    ref2 = set(m2)

    assert measurements == ref1 | ref2


def test_add_duplicate():
    # adding, but the two maps have a duplicate measurement
    m1 = cinnabar.FEMap()
    m1.add_experimental_measurement(
        label="c1",
        value=10.1 * unit.nanomolar,
        uncertainty=0.2 * unit.nanomolar,
    )
    m1.add_experimental_measurement(
        label="c2",
        value=10.2 * unit.nanomolar,
        uncertainty=0.3 * unit.nanomolar,
    )

    m2 = cinnabar.FEMap()
    m2.add_experimental_measurement(
        label="c1",
        value=10.1 * unit.nanomolar,
        uncertainty=0.2 * unit.nanomolar,
    )
    m2.add_absolute_calculation(
        label="c1",
        value=-9.5 * unit.kilocalorie_per_mole,
        uncertainty=0.4 * unit.kilocalorie_per_mole,
    )

    m3 = m1 + m2

    assert len(m3) == 3
    measurements = set(m3)

    ref1 = set(m1)
    ref2 = set(m2)

    assert measurements == ref1 | ref2
