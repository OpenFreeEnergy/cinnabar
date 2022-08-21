import itertools

import networkx as nx
import numpy as np
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
        assert (
            diff < covar[i, i]
        ), f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."


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
        assert (
            diff < covar[i, i]
        ), f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."


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
        assert (
            diff < covar[i, i]
        ), f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."


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

        assert (
            np.abs(true_diff - mle_diff) < 1.0
        ), f"Relative\
         difference from MLE: {mle_diff} is too far from the\
         input difference, {true_diff}"


def test_correlation_positive(fe_map):
    """
    Test that the absolute DG plots have the correct signs,
    and statistics within reasonable agreement to the example data
    in `cinnabar/data/example.csv`
    """

    nodes = fe_map.graph.nodes

    x_data = np.asarray([n[1]["exp_DG"] for n in nodes(data=True)])
    y_data = np.asarray([n[1]["calc_DG"] for n in nodes(data=True)])
    xerr = np.asarray([n[1]["exp_dDG"] for n in nodes(data=True)])
    yerr = np.asarray([n[1]["calc_dDG"] for n in nodes(data=True)])

    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="rho")
    assert 0 < bss["mle"] < 1, "Correlation must be positive for this data"

    for stat in ["R2", "rho"]:
        bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
        # all of the statistics for this example is between 0.61 and 0.84
        assert (
            0.5 < bss["mle"] < 0.9
        ), f"Correlation must be positive for this data. {stat} is {bss['mle']}"
