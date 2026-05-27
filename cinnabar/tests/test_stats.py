import itertools

import networkx as nx
import numpy as np
import pytest

from cinnabar.estimators import MLEEstimator
from cinnabar.stats import bootstrap_statistic



def test_correlation_positive(example_data_mle):
    """
    Test that the absolute DG plots have the correct signs,
    and statistics within reasonable agreement to the example data
    in `cinnabar/data/example.csv`
    """
    x_data, y_data, xerr, yerr = example_data_mle

    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="rho")
    assert 0 < bss["mle"] < 1, "Correlation must be positive for this data"

    for stat in ["R2", "rho", "KTAU"]:
        bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
        # all of the statistics for this example is between 0.61 and 0.84
        assert 0.5 < bss["mle"] < 0.9, f"Correlation must be positive for this data. {stat} is {bss['mle']}"


def test_missing_statistic(example_data_mle):
    """
    Test that an error is raised when an unknown statistic is requested
    """
    x_data, y_data, xerr, yerr = example_data_mle

    with pytest.raises(ValueError, match="unknown statistic UNKNOWN_STAT"):
        bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="UNKNOWN_STAT")


def test_inconsistent_array_shape():
    """
    Test that an error is raised when input arrays have inconsistent shapes
    """
    x_data = np.array([1.0, 2.0, 3.0])
    y_data = np.array([1.0, 2.0])  # inconsistent shape
    xerr = np.array([0.1, 0.1, 0.1])
    yerr = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="All input arrays must have the same length"):
        bootstrap_statistic(x_data, y_data, xerr, yerr, statistic="RMSE")


def test_confidence_intervals_defaults(example_data_mle):
    """
    Test that boostrap confidence intervals contains
    the 'mle' value when using defaults.
    """
    error_message = "The stat must lie within the bootstrapped 95% CI"
    x_data, y_data, xerr, yerr = example_data_mle
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
def test_confidence_intervals(example_data_mle, stat, true_uncert, pred_uncert, estimate):
    """
    Test that the bootstrapped confidence intervals contain the
    corresponding statistics.
    Uses the example data in `cinnabar/data/example.csv`
    """
    error_message = "The stat must lie within the bootstrapped 95% CI"
    x_data, y_data, xerr, yerr = example_data_mle
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
        ("NRMSE", 1.0040857354711985),
        ("PI", 0.816249795651462),
    ],
)
def test_regression_bootstrap_statistics(example_data_mle, stat, expected):
    """
    Regression test for bootstrap statistics on example data
    in `cinnabar/data/example.csv`
    """
    x_data, y_data, xerr, yerr = example_data_mle

    bss = bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
    assert pytest.approx(bss["mle"], rel=1e-6) == expected, f"Regression test failed for statistic {stat}"
    error_message = "The stat must lie within the bootstrapped 95% CI"
    assert (bss["low"] < bss["mle"]) and (bss["mle"] < bss["high"]), error_message


def test_bootstrap_statistic_no_errors(example_data_mle):
    """
    Test that compute_statistic works when no errors are provided
    """
    x_data, y_data, _, _ = example_data_mle

    bss = bootstrap_statistic(x_data, y_data, statistic="RMSE")
    assert pytest.approx(bss["mle"], rel=1e-6) == 9.364494046790412


def test_bad_edge_matrix_action(fe_map):
    """
    Test that an error is raised when an unknown action is provided
    to the edge matrix computation
    """
    with pytest.raises(ValueError, match='action "bad_action" unknown'):
        _ = MLEEstimator.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="bad_action")


def test_edge_matrix_no_action(fe_map):
    edge_matrix = MLEEstimator.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action=None)
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)


def test_edge_matrix_symmetrize(fe_map):
    edge_matrix = MLEEstimator.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="symmetrize")
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)
    # check the matrix is symmetric
    assert np.allclose(edge_matrix, edge_matrix.T)


def test_edge_matrix_antisymmetrize(fe_map):
    edge_matrix = MLEEstimator.form_edge_matrix(fe_map.to_legacy_graph(), label="calc_DDG", action="antisymmetrize")
    assert edge_matrix.shape == (fe_map.n_ligands, fe_map.n_ligands)
    # check the matrix is antisymmetric
    assert np.allclose(edge_matrix, -edge_matrix.T)
