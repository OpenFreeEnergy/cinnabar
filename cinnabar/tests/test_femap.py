import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from openff.units import unit

import cinnabar
from cinnabar import estimators, femap


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


def test_eq_wrong_type():
    m1 = cinnabar.FEMap()
    m2 = "not a FEMap"

    assert (m1 == m2) is False


def test_degree(example_map):
    assert example_map.degree == pytest.approx(58 / 36)


def test_n_measurements(example_map):
    assert example_map.n_measurements == 94


def test_weakly_connected(example_map):
    assert example_map.check_weakly_connected() is True


def test_weakly_connected_no_results():
    m = cinnabar.FEMap()
    with pytest.raises(ValueError, match="Graph contains no computational edges, cannot check connectivity"):
        m.check_weakly_connected()


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

        assert y.magnitude == pytest.approx(y_ref), e
        assert yerr.magnitude == pytest.approx(yerr_ref), e
    # check the metadata is correct
    metadata = example_map.get_estimator_metadata("MLE")
    # check general metadata is correct
    assert metadata.source == "MLE"
    assert metadata.estimator == "MLEEstimator"
    # check mle specific metadata is correct
    assert len(metadata.ligand_order) == len(ref_mle_results)
    assert metadata.covariance_matrix.shape == (len(ref_mle_results), len(ref_mle_results))


def test_generate_absolute_values_multiple_sources(example_map, ref_mle_results):
    # add a second set of measurements with a different source
    # these are the same values as the original with 0.25 kcal/mol added to each, but with a different source
    to_add = []
    for m in example_map:
        if m.computational:
            to_add.append(
                cinnabar.Measurement(
                    labelA=m.labelA,
                    labelB=m.labelB,
                    DG=m.DG + 0.25 * unit.kilocalorie_per_mole,
                    uncertainty=m.uncertainty,
                    computational=True,
                    source="other_source",
                )
            )
    for m in to_add:
        example_map.add_measurement(m)
    # generate the values with the MLE estimator
    example_map.generate_absolute_values(estimator=estimators.MLEEstimator())
    # we should have two sets of absolute values, one for each source
    abs_df = example_map.get_absolute_dataframe()
    assert abs_df.shape == (108, 5)
    # we should have two sets of metadata, one for each source
    for source in ("MLE()", "MLE(other_source)"):
        meta = example_map.get_estimator_metadata(source)
        assert meta.source == source
        assert meta.estimator == "MLEEstimator"
        assert meta.covariance_matrix.shape == (36, 36)
        assert len(meta.ligand_order) == 36

    # make sure the reference values for the original source are the same as before
    for e, (y_ref, yerr_ref) in ref_mle_results.items():
        # get the calculated DG from the dataframe
        calculated_dg = abs_df.loc[(abs_df.label == e) & (abs_df.computational) & (abs_df.source == "MLE()")]
        assert calculated_dg["DG (kcal/mol)"].values[0] == pytest.approx(y_ref)
        assert calculated_dg["uncertainty (kcal/mol)"].values[0] == pytest.approx(yerr_ref)


def test_generate_absolute_values_no_results():
    m = cinnabar.FEMap()

    with pytest.raises(ValueError, match="FEMap contains no measurements"):
        m.generate_absolute_values()


def test_generate_absolute_values_not_connected():
    m = cinnabar.FEMap()

    m.add_relative_calculation(
        labelA="ligA",
        labelB="ligB",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
        source="test",
    )
    m.add_relative_calculation(
        labelA="ligC",
        labelB="ligD",
        value=-2.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
        source="test",
    )
    with pytest.raises(ValueError, match="Computational results for source 'test' are not fully connected"):
        m.generate_absolute_values()


def test_generate_absolute_values_mixed_units():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "ligA",
        "ligB",
        DG=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    graph.add_edge(
        "ligB",
        "ligC",
        DG=-4.0 * unit.kilojoule_per_mole,
        uncertainty=0.2 * unit.kilojoule_per_mole,
    )
    m = cinnabar.FEMap.from_networkx(graph)
    with pytest.raises(ValueError, match="All units must be the same"):
        m.generate_absolute_values()


def test_generate_absolute_values_repeats():
    """Make sure an error is raised if there are multiple edges between same nodes and we try and use the MLE solver."""
    fe_map = femap.FEMap()
    fe_map.add_relative_calculation(
        "ligA",
        "ligB",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    # add a repeated edge
    fe_map.add_relative_calculation(
        "ligA",
        "ligB",
        value=-1.2 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
    )
    fe_map.add_relative_calculation(
        "ligB",
        "ligC",
        value=-4.0 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
    )
    with pytest.raises(ValueError, match="Multiple edges detected between nodes ligA and ligB."):
        fe_map.generate_absolute_values()


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


def test_add_wrong_type():
    m1 = cinnabar.FEMap()
    m2 = "not a FEMap"

    with pytest.raises(TypeError, match="unsupported operand"):
        _ = m1 + m2


def test_draw_graph_to_file(fe_map, tmp_path):
    filepath = tmp_path / "femap_graph.png"
    fe_map.draw_graph(title="test", filename=filepath)

    assert filepath.exists()


def test_draw_graph_show(fe_map, monkeypatch):
    called = {}

    def mock_show():
        called["show"] = True

    monkeypatch.setattr(plt, "show", mock_show)

    fe_map.draw_graph(title="test", filename=None)

    assert called.get("show", False) is True


def test_to_legacy_missing_exp():
    """Check we can convert to legacy graph when no experimental data is present"""
    m = cinnabar.FEMap()

    m.add_relative_calculation(
        labelA="ligA",
        labelB="ligB",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    g = m.to_legacy_graph()

    assert isinstance(g, nx.DiGraph)


def test_to_legacy_not_connected():
    """Check we can convert to legacy graph when graph is not connected"""
    m = cinnabar.FEMap()

    m.add_relative_calculation(
        labelA="ligA",
        labelB="ligB",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    m.add_relative_calculation(
        labelA="ligC",
        labelB="ligD",
        value=-2.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    with pytest.warns(UserWarning, match="Graph is not connected enough to compute absolute values"):
        g = m.to_legacy_graph()

        assert isinstance(g, nx.DiGraph)


def test_measurement_ordering(example_map):
    """Check that the ordering of the measurements does not change the result of the MLE solver."""
    # generate a new map with edges added in a random order
    rng = np.random.default_rng()
    measurements = list(example_map)
    rng.shuffle(measurements)  # generate a random order of the edges
    femap2 = cinnabar.FEMap()
    for m in measurements:
        femap2.add_measurement(m)

    # generate the reference values
    example_map.generate_absolute_values()
    abs_df = example_map.get_absolute_dataframe()
    abs_df = abs_df.loc[abs_df.computational].copy().reset_index(drop=True)  # just grab the computational values
    # generate the new values using the random ordering
    femap2.generate_absolute_values()
    abs_df2 = femap2.get_absolute_dataframe()
    abs_df2 = abs_df2.loc[abs_df2.computational].copy().reset_index(drop=True)
    # check the results are the same after aligning on the ligand name
    abs_df = abs_df.sort_values("label").reset_index(drop=True)
    abs_df2 = abs_df2.sort_values("label").reset_index(drop=True)

    assert np.allclose(abs_df["DG (kcal/mol)"].values, abs_df2["DG (kcal/mol)"].values)
