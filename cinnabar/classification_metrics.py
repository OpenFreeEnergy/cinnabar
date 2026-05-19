# This code is part of cinnabar and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/cinnabar

import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _create_2d_histogram(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[NDArray, NDArray, NDArray]:
    """
    Create a 2D histogram from two arrays of data.

    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.

    Returns
    -------
    histogram : ndarray
        The 2D histogram of the input data.
    bins_true : ndarray
        The bin edges along the y_true axis.
    bins_pred : ndarray
        The bin edges along the y_pred axis.

    Raises
    ------
    ValueError
        If `y_true` and `y_pred` have different lengths.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    y_true_sorted = np.sort(y_true)
    y_pred_sorted = np.sort(y_pred)
    # Calculate bin edges using midpoints between sorted values
    bins_true = np.concatenate(([y_true.min()], (y_true_sorted[:-1] + y_true_sorted[1:]) / 2, [y_true.max()]))
    bins_pred = np.concatenate(([y_pred.min()], (y_pred_sorted[:-1] + y_pred_sorted[1:]) / 2, [y_pred.max()]))

    # Note a perfect prediction will have all counts in the diagonal bins
    histogram, bins_true, bins_pred = np.histogram2d(y_true, y_pred, bins=[bins_true, bins_pred])

    return histogram, bins_true, bins_pred


def _compute_overlap_coefficient(histogram: NDArray, ranking: int) -> float:
    """
    Compute the overlap coefficient from a 2D histogram.

    The overlap coefficient is calculated based on the counts in the histogram
    for the top N ranked ligands (most active).

    Parameters
    ----------
    histogram : ndarray
        A 2D histogram array where the counts are stored.
    ranking : int
        The number of rankings to consider when computing overlap.

    Returns
    -------
    float
        The overlap coefficient.

    Raises
    ------
    ValueError
        If `top_n_ligands` is greater than the number of ligands in the histogram.
    """
    if ranking < 1:
        raise ValueError("Ranking must be greater than 0.")

    if histogram.shape[0] < ranking:
        raise ValueError("Ranking must be less than the number of ligands.")

    overlap = np.sum(histogram[:ranking, :ranking])

    return overlap / ranking


def compute_fraction_best_ligands(y_true: Iterable[float], y_pred: Iterable[float], fraction: float = 0.5) -> float:
    """
    Compute the fraction of the best ligands metric introduced by Chris Bayly.

    This function calculates the fraction of the best ligands by computing overlap
    coefficients for each ranking up to the number of ligands and then averaging up to the specified fraction.

    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.
    fraction : float
        The fraction of ligands to consider as the best (default is 0.5).

    Returns
    -------
    float
        The computed fraction of the best ligands.

    Raises
    ------
    ValueError
        If `fraction` is not between 0 and 1.
    """

    if not (0 <= fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")

    histogram = _create_2d_histogram(y_true, y_pred)[0]

    num_ligands = histogram.shape[0]
    num_best_ligands = math.floor(num_ligands * fraction)

    overlap_coefficients = [_compute_overlap_coefficient(histogram, i + 1) for i in range(num_best_ligands)]

    fraction_best_ligands = sum(overlap_coefficients) / num_best_ligands

    return fraction_best_ligands
