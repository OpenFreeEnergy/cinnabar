import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from openff.units import unit

import cinnabar
from cinnabar import conversion, estimators, femap, stats


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
    ref_v = -9.58 * unit.kilocalorie_per_mole
    ref_u = 0.06 * unit.kilocalorie_per_mole
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
    assert isinstance(metadata, estimators.MLEEstimatorResult)
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


@pytest.mark.parametrize(
    "dataframe_func, expected",
    [
        pytest.param(lambda fe: fe.get_relative_dataframe(), 1.05, id="relative"),
        pytest.param(lambda fe: fe.get_all_to_all_relative_dataframe(symmetrical=False), 1.21, id="all-to-all"),
    ],
)
def test_to_relative_dataframe_regression(example_map, dataframe_func, expected):
    """Test that the values in the relative/pairwise relative dataframe match the original values in the map."""
    # needed for the pairwise dataframe
    example_map.generate_absolute_values()
    rel_df = dataframe_func(example_map)
    # calculate the RMSE between the reported edges and the experimental differences
    # make sure this matches some historic data
    calc_ddg = rel_df[rel_df["computational"] == True]["DDG (kcal/mol)"].values
    exp_ddg = rel_df[rel_df["computational"] == False]["DDG (kcal/mol)"].values
    rmse = stats.calculate_rmse(exp_ddg, calc_ddg)
    assert rmse == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "dataframe_func",
    [
        pytest.param(lambda fe, observable: fe.get_relative_dataframe(observable_type=observable), id="relative"),
        pytest.param(
            # we need the abs values for the all-to-all method
            lambda fe, observable: fe.get_all_to_all_relative_dataframe(symmetrical=False, observable_type=observable),
            id="all-to-all",
        ),
    ],
)
def test_to_relative_dataframe_pic50(example_map, dataframe_func):
    """Test that returning the values in units of pIC50 gives error metrics consistent with the values in kcal/mol"""
    example_map.generate_absolute_values()
    rel_dg = dataframe_func(example_map, "dg")
    rel_pci50 = dataframe_func(example_map, "pic50")
    assert rel_dg.shape == rel_pci50.shape

    # make sure the column order is the same but the names have been updated
    assert rel_pci50.columns.tolist() == [
        "labelA",
        "labelB",
        "ΔpIC50",
        "uncertainty (ΔpIC50)",
        "source",
        "computational",
    ]

    # as dg to pic50 is linear check the RMSE also converts
    calc_ddg = rel_dg[rel_dg["computational"] == True]["DDG (kcal/mol)"].values
    exp_ddg = rel_dg[rel_dg["computational"] == False]["DDG (kcal/mol)"].values
    rmse_dg = stats.calculate_rmse(exp_ddg, calc_ddg)
    # same again for pic50
    calc_dpic50 = rel_pci50[rel_pci50["computational"] == True]["ΔpIC50"].values
    exp_dpic50 = rel_pci50[rel_pci50["computational"] == False]["ΔpIC50"].values
    rmse_pic50 = stats.calculate_rmse(exp_dpic50, calc_dpic50)
    # convert the error back
    rmse_pic50, _ = conversion.convert_observable(rmse_pic50, "pic50", "dg")
    # we need to use the abs value due to the conversion
    assert rmse_dg == pytest.approx(abs(rmse_pic50), abs=0.01)


def test_to_absolute_dataframe_regression(example_map):
    """Test that the values in the absolute dataframe match the original values in the map."""
    example_map.generate_absolute_values()
    abs_df = example_map.get_absolute_dataframe()
    # make sure the RMSE and R2 matches some historic data
    calc_dg = abs_df[abs_df["computational"] == True]["DG (kcal/mol)"].to_numpy(copy=True)
    exp_dg = abs_df[abs_df["computational"] == False]["DG (kcal/mol)"].to_numpy(copy=True)
    # as the MLE values are centered at zero apply a shift to the experimental mean
    calc_dg -= calc_dg.mean()  # remove any systematic shift
    calc_dg += exp_dg.mean()
    rmse = stats.calculate_rmse(exp_dg, calc_dg)
    assert rmse == pytest.approx(0.84, abs=0.01)
    r_2 = stats.calculate_r2(exp_dg, calc_dg)
    assert r_2 == pytest.approx(0.61, abs=0.01)


def test_to_absolute_dataframe_pic50(example_map):
    """Test that returning the values in units of pIC50 gives error metrics consistent with the values in kcal/mol"""
    example_map.generate_absolute_values()
    abs_dg = example_map.get_absolute_dataframe()
    abs_pci50 = example_map.get_absolute_dataframe(observable_type="pic50")
    assert abs_dg.shape == abs_pci50.shape

    # make sure the column order is the same but the names have been updated
    assert abs_pci50.columns.tolist() == ["label", "pIC50", "uncertainty (pIC50)", "source", "computational"]

    # as dg to pic50 is linear check the RMSE also converts
    calc_ddg = abs_dg[abs_dg["computational"] == True]["DG (kcal/mol)"].values
    exp_ddg = abs_dg[abs_dg["computational"] == False]["DG (kcal/mol)"].values
    rmse_dg = stats.calculate_rmse(exp_ddg, calc_ddg)
    # same again for pic50
    calc_dpic50 = abs_pci50[abs_pci50["computational"] == True]["pIC50"].values
    exp_dpic50 = abs_pci50[abs_pci50["computational"] == False]["pIC50"].values
    rmse_pic50 = stats.calculate_rmse(exp_dpic50, calc_dpic50)
    # convert the error back
    rmse_pic50, _ = conversion.convert_observable(rmse_pic50, "pic50", "dg")
    # we need to use the abs value due to the conversion
    assert rmse_dg == pytest.approx(abs(rmse_pic50), abs=0.01)  # use abs 0.01 as we round pic50 to 2dp
    # also check the ranking
    r_2_dg = stats.calculate_r2(exp_ddg, calc_ddg)
    r_2_pic50 = stats.calculate_r2(exp_dpic50, calc_dpic50)
    assert r_2_dg == pytest.approx(r_2_pic50, abs=0.01)  # use abs 0.01 as we round pic50 to 2dp


def test_to_dataframe(example_map):
    abs_df = example_map.get_absolute_dataframe()
    rel_df = example_map.get_relative_dataframe()

    assert abs_df.shape == (36, 5)
    # the dataframe should have the simulated and experimental values
    assert rel_df.shape == (116, 6)
    # check the split between the results is correct
    assert rel_df.loc[rel_df.computational].shape == (58, 6)
    assert rel_df.loc[~rel_df.computational].shape == (58, 6)

    example_map.generate_absolute_values()

    abs_df2 = example_map.get_absolute_dataframe()
    assert abs_df2.shape == (72, 5)
    assert abs_df2.loc[abs_df2.computational].shape == (36, 5)
    assert abs_df2.loc[~abs_df2.computational].shape == (36, 5)


def test_to_all_pairwise_df_symmetry(example_map):
    """Test generating the all-to-all pairwise dataframe with the symmetry option."""
    # Generate using only the experimental data
    all_symmetry_df = example_map.get_all_to_all_relative_dataframe(symmetrical=True)
    all_no_sym_df = example_map.get_all_to_all_relative_dataframe(symmetrical=False)

    assert all_symmetry_df.shape == (36 * 35, 6)  # n(n-1) for symmetrical
    assert all_no_sym_df.shape == (36 * 35 // 2, 6)  # n(n-1) / 2 for non-symmetrical

    # generate again using the calculated absolute values as well
    example_map.generate_absolute_values()
    all_symmetry_df2 = example_map.get_all_to_all_relative_dataframe(symmetrical=True)
    all_no_sym_df2 = example_map.get_all_to_all_relative_dataframe(symmetrical=False)

    assert all_symmetry_df2.shape == (36 * 35 * 2, 6)  # n(n-1) for symmetrical * 2 for both sources
    assert all_no_sym_df2.shape == (36 * 35, 6)  # n(n-1) / 2 for non-symmetrical * 2 for both sources

    # make sure the pair ordering is the same for each source
    for df in [all_symmetry_df2, all_no_sym_df2]:
        # get a 2D array of the label pairs for each source and check they are the same
        comp_labels = df.loc[df.computational, ["labelA", "labelB"]].values
        exp_labels = df.loc[~df.computational, ["labelA", "labelB"]].values
        assert (comp_labels == exp_labels).all()

    # make sure all values are correct
    abs_df = example_map.get_absolute_dataframe()
    for df in [all_symmetry_df2, all_no_sym_df2]:
        for _, row in df.iterrows():
            label_a = row["labelA"]
            label_b = row["labelB"]
            computational = row["computational"]

            if computational:
                source_df = abs_df.loc[abs_df.computational]
            else:
                source_df = abs_df.loc[~abs_df.computational]

            dg_a = source_df.loc[source_df.label == label_a, "DG (kcal/mol)"].values[0]
            dg_b = source_df.loc[source_df.label == label_b, "DG (kcal/mol)"].values[0]
            expected_ddg = dg_b - dg_a
            assert row["DDG (kcal/mol)"] == expected_ddg


def test_all_to_all_pairwise_df_absolute(example_map):
    """Test that we can generate the all-to-all pairwise dataframe using the absolute values,
    and that it matches the pairwise df generated from the original map.
    """
    example_map.generate_absolute_values()
    abs_df = example_map.get_absolute_dataframe()
    abs_df = abs_df[(abs_df["computational"] == True) & (abs_df["source"] == "MLE")]
    pair_rel_df = example_map.get_all_to_all_relative_dataframe(symmetrical=False)
    pair_rel_df = pair_rel_df[pair_rel_df["computational"] == True].reset_index(drop=True)

    abs_map = femap.FEMap()
    # add each predicted absolute value as a measurement and compute the pairwise df again
    for _, row in abs_df.iterrows():
        abs_map.add_absolute_calculation(
            label=row["label"],
            value=row["DG (kcal/mol)"] * unit.kilocalorie_per_mole,
            uncertainty=row["uncertainty (kcal/mol)"] * unit.kilocalorie_per_mole,
            source="ABFE",
        )

    abs_pair_rel_df = abs_map.get_all_to_all_relative_dataframe(symmetrical=False)
    # the df should match between the two maps, except for the uncertainty and source columns which will be different
    # due to the different sources and the way uncertainties are propagated for the pairwise df
    pd.testing.assert_frame_equal(
        pair_rel_df.drop(columns=["uncertainty (kcal/mol)", "source"]),
        abs_pair_rel_df.drop(columns=["uncertainty (kcal/mol)", "source"]),
    )
    # we should also check that the uncertainty is different as we should be using the covariance in the pair_rel_df
    assert not np.array_equal(
        pair_rel_df["uncertainty (kcal/mol)"].values, abs_pair_rel_df["uncertainty (kcal/mol)"].values
    )


def test_all_to_all_pairwise_df_no_data():
    """Test that we can generate the all-to-all pairwise dataframe with no data without error."""
    fe_map = femap.FEMap()
    df = fe_map.get_all_to_all_relative_dataframe(symmetrical=False)
    assert len(df) == 0
    # make sure the columns are still correct though
    assert df.columns.tolist() == [
        "labelA",
        "labelB",
        "DDG (kcal/mol)",
        "uncertainty (kcal/mol)",
        "source",
        "computational",
    ]


def test_to_all_pairwise_df_uses_covariance_matrix():
    fe_map = cinnabar.FEMap()
    kjpm = unit.kilojoule_per_mole

    fe_map.add_relative_calculation("A", "B", 4.184 * kjpm, 0.4184 * kjpm)
    fe_map.add_relative_calculation("B", "C", 8.368 * kjpm, 0.8368 * kjpm)
    fe_map.add_relative_calculation("A", "C", 12.552 * kjpm, 1.2552 * kjpm)

    fe_map.generate_absolute_values()

    pairwise_df = fe_map.get_all_to_all_relative_dataframe(symmetrical=False)
    abs_df = fe_map.get_absolute_dataframe().set_index("label")
    metadata = fe_map.get_estimator_metadata("MLE")
    assert isinstance(metadata, estimators.MLEEstimatorResult)
    label_to_index = {label: i for i, label in enumerate(metadata.ligand_order)}

    for _, row in pairwise_df.iterrows():
        label_a = row["labelA"]
        label_b = row["labelB"]
        i = label_to_index[label_a]
        j = label_to_index[label_b]
        covariance = metadata.covariance_matrix[i, j]
        expected_uncertainty = (
            abs_df.loc[label_a, "uncertainty (kcal/mol)"] ** 2
            + abs_df.loc[label_b, "uncertainty (kcal/mol)"] ** 2
            - 2 * covariance
        ) ** 0.5

        assert row["uncertainty (kcal/mol)"] == pytest.approx(expected_uncertainty)

    naive_uncertainty = (
        abs_df.loc["A", "uncertainty (kcal/mol)"] ** 2 + abs_df.loc["B", "uncertainty (kcal/mol)"] ** 2
    ) ** 0.5
    ab_uncertainty = pairwise_df.loc[
        (pairwise_df.labelA == "A") & (pairwise_df.labelB == "B"), "uncertainty (kcal/mol)"
    ].iloc[0]
    assert ab_uncertainty < naive_uncertainty


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


def test_missing_estimator_metadata(example_map):
    with pytest.raises(KeyError, match="No estimator metadata stored for source test."):
        example_map.generate_absolute_values()
        example_map.get_estimator_metadata("test")
