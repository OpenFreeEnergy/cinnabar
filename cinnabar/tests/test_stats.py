import itertools

import networkx as nx
import numpy as np
import pytest

from cinnabar import stats
from cinnabar.stats import bootstrap_statistic


def test_mle_easy():
    """
    Test that the MLE for a graph with an absolute
    estimate on all nodes will recapitulate it
    """
    input_absolutes: list = [-14.0, -13.0, -9.0]
    graph = nx.DiGraph()
    for i, val in enumerate(input_absolutes):
        graph.add_node(i, f_i=val, f_di=0.5)

    edges = [(0, 1), (0, 2), (2, 1)]
    for node1, node2 in edges:
        noise = np.random.uniform(low=-1.0, high=1.0)
        diff = input_absolutes[node2] - input_absolutes[node1] + noise
        graph.add_edge(node1, node2, f_ij=diff, f_dij=0.5 + np.abs(noise))

    output_absolutes, covar = stats.mle(graph, factor="f_ij", node_factor="f_i")

    for i, _ in enumerate(graph.nodes(data=True)):
        diff = np.abs(output_absolutes[i] - input_absolutes[i])
        assert diff < covar[i, i], (
            f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."
        )


def test_mle_easy_self_edge():
    """
    Test that the MLE for a graph with an absolute
    estimate on all nodes will recapitulate it
    when a self-edge is included
    """
    input_absolutes: list = [-14.0, -13.0, -9.0]
    graph = nx.DiGraph()
    for i, val in enumerate(input_absolutes):
        graph.add_node(i, f_i=val, f_di=0.5)

    edges = [(0, 1), (0, 2), (2, 1), (0, 0)]
    for node1, node2 in edges:
        noise = np.random.uniform(low=-1.0, high=1.0)
        diff = input_absolutes[node2] - input_absolutes[node1] + noise
        graph.add_edge(node1, node2, f_ij=diff, f_dij=0.5 + np.abs(noise))

    output_absolutes, covar = stats.mle(graph, factor="f_ij", node_factor="f_i")

    for i, _ in enumerate(graph.nodes(data=True)):
        diff = np.abs(output_absolutes[i] - input_absolutes[i])
        assert diff < covar[i, i], (
            f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."
        )


def test_mle_hard():
    """
    Test that the MLE for a graph with a node missing an absolute value
    can get it right based on relative results
    """
    input_absolutes: list = [-14.0, -13.0, -9.0]
    # make a t
    graph = nx.DiGraph()
    # Don't assign the first absolute value, check that MLE can get close to it
    for i, val in enumerate(input_absolutes):
        if i == 0:
            graph.add_node(i)
        else:
            graph.add_node(i, f_i=val, f_di=0.5)

    edges = [(0, 1), (0, 2), (2, 1)]
    for node1, node2 in edges:
        noise = np.random.uniform(low=-1.0, high=1.0)
        diff = input_absolutes[node2] - input_absolutes[node1] + noise
        graph.add_edge(node1, node2, f_ij=diff, f_dij=0.5 + np.abs(noise))

    output_absolutes, covar = stats.mle(graph, factor="f_ij", node_factor="f_i")

    for i, _ in enumerate(graph.nodes(data=True)):
        diff = np.abs(output_absolutes[i] - input_absolutes[i])
        assert diff < covar[i, i], (
            f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."
        )


def test_mle_relative():
    """
    Test that the MLE can get the relative differences correct
     when no absolute values are provided
    """
    input_absolutes: list = [-14.0, -13.0, -9.0]
    graph = nx.DiGraph()
    # Don't assign any absolute values
    edges = [(0, 1), (0, 2), (2, 1)]
    for node1, node2 in edges:
        noise = np.random.uniform(low=-0.5, high=0.5)
        diff = input_absolutes[node2] - input_absolutes[node1] + noise
        graph.add_edge(node1, node2, f_ij=diff, f_dij=0.5 + np.abs(noise))

    output_absolutes, _ = stats.mle(graph, factor="f_ij", node_factor="f_i")

    pairs = itertools.combinations(range(len(input_absolutes)), 2)

    for i, j in pairs:
        mle_diff = output_absolutes[i] - output_absolutes[j]
        true_diff = input_absolutes[i] - input_absolutes[j]

        assert np.abs(true_diff - mle_diff) < 1.0, (
            f"Relative\
         difference from MLE: {mle_diff} is too far from the\
         input difference, {true_diff}"
        )


def test_mle_bidirectional_edges():
    """Make sure an error is raised if there are repeated edges in the graph."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, f_ij=1.0, f_dij=0.5)
    graph.add_edge(1, 0, f_ij=-2.0, f_dij=0.5)  # repeated edge

    with pytest.raises(ValueError, match="Multiple edges detected between nodes 1 and 0."):
        stats.mle(graph, factor="f_ij", node_factor="f_i")



def test_correlation_positive(example_data):
    """
    Test that the absolute DG plots have the correct signs,
    and statistics within reasonable agreement to the example data
    in `cinnabar/data/example.csv`
    """
    x_data, y_data, xerr, yerr = example_data

    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="rho")
    assert 0 < bss["mle"] < 1, "Correlation must be positive for this data"

    for stat in ["R2", "rho", "KTAU"]:
        bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
        # all of the statistics for this example is between 0.61 and 0.84
        assert 0.5 < bss["mle"] < 0.9, f"Correlation must be positive for this data. {stat} is {bss['mle']}"


def test_missing_statistic(example_data):
    """
    Test that an error is raised when an unknown statistic is requested
    """
    x_data, y_data, xerr, yerr = example_data

    with pytest.raises(Exception, match="unknown statistic UNKNOWN_STAT"):
        bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="UNKNOWN_STAT")


def test_confidence_intervals_defaults(example_data):
    """
    Test that boostrap confidence intervals contains
    the 'mle' value when using defaults.
    """
    error_message = "The stat must lie within the bootstrapped 95% CI"
    x_data, y_data, xerr, yerr = example_data
    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="RMSE")
    assert bss["low"] < bss["mle"] < bss["high"], error_message


@pytest.mark.parametrize(
    "stat,true_uncert,pred_uncert,estimate",
    [
        ["RMSE", False, False, "mle"],
        ["MUE", False, False, "mle"],
        ["RMSE", True, False, "mean"],
        ["RMSE", False, True, "mean"],
        ["RMSE", True, True, "mean"],
    ],
)
def test_confidence_intervals(example_data, stat, true_uncert, pred_uncert, estimate):
    """
    Test that the bootstrapped confidence intervals contain the
    corresponding statistics.
    Uses the example data in `cinnabar/data/example.csv`
    """
    error_message = "The stat must lie within the bootstrapped 95% CI"
    x_data, y_data, xerr, yerr = example_data
    bss = bootstrap_statistic(
        x_data,
        y_data,
        xerr,
        yerr,
        statistic=stat,
        include_true_uncertainty=true_uncert,
        include_pred_uncertainty=pred_uncert,
    )
    assert bss["low"] < bss[estimate] < bss["high"], error_message


def test_confidence_interval_edge_case():
    """
    Test that the bootstrapped confidence interval
    for RMSE contains the sample estimate.
    Uses the data from https://github.com/OpenFreeEnergy/cinnabar/issues/73
    """

    # Data from Cinnabar issue #73
    x_data = np.array([-0.101, 0.351, 0.117, 0.623, 5.172, 5.209, -1.727, -1.387, -1.534, 1.082])
    y_data = np.array([-0.174, 0.42, 0.262, 0.626, 5.064, 4.783, -1.58, -1.712, -1.699, 0.822])
    xerr = np.array([0.443, 0.652, 0.57, 0.245, 1.112, 1.049, 1.23, 1.435, 1.521, 0.505])
    yerr = np.array([0.442, 0.714, 0.619, 0.224, 1.401, 1.107, 1.178, 1.252, 1.265, 0.472])

    # RMSE (default mode)
    bss = bootstrap_statistic(
        x_data, y_data, xerr, yerr, statistic="RMSE", include_true_uncertainty=False, include_pred_uncertainty=False
    )
    error_message = "The stat must lie within the bootstrapped 95% CI"
    assert (bss["low"] < bss["mle"]) and (bss["mle"] < bss["high"]), error_message


@pytest.mark.parametrize(
    "stat, expected",
    [
        ("RMSE", 9.364494046790412),
        ("MUE", 9.326388888888888),
        ("R2", 0.6149662203714674),
        ("rho", 0.7841978196676316),
        ("KTAU", 0.58148151940828),
        ("RAE", 15.995712243925674),
    ],
)
def test_regression_bootstrap_statistics(example_data, stat, expected):
    """
    Regression test for bootstrap statistics on example data
    in `cinnabar/data/example.csv`
    """
    x_data, y_data, xerr, yerr = example_data

    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
    assert pytest.approx(bss["mle"], rel=1e-6) == expected, f"Regression test failed for statistic {stat}"
    error_message = "The stat must lie within the bootstrapped 95% CI"
    assert (bss["low"] < bss["mle"]) and (bss["mle"] < bss["high"]), error_message


def test_bootstrap_statistic_no_errors(example_data):
    """
    Test that compute_statistic works when no errors are provided
    """
    x_data, y_data, _, _ = example_data

    bss = bootstrap_statistic(x_data, y_data, statistic="RMSE")
    assert pytest.approx(bss["mle"], rel=1e-6) == 9.364494046790412


def test_bad_edge_matrix_action(fe_map):
    """
    Test that an error is raised when an unknown action is provided
    to the edge matrix computation
    """
    with pytest.raises(Exception, match='action "bad_action" unknown'):
        _ = stats.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="bad_action")


def test_edge_matrix_no_action(fe_map):
    edge_matrix = stats.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action=None)
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)


def test_edge_matrix_symmetrize(fe_map):
    edge_matrix = stats.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="symmetrize")
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)
    # check the matrix is symmetric
    assert np.allclose(edge_matrix, edge_matrix.T)


def test_edge_matrix_antisymmetrize(fe_map):
    edge_matrix = stats.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="antisymmetrize")
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)
    # check the matrix is antisymmetric
    assert np.allclose(edge_matrix, -edge_matrix.T)
