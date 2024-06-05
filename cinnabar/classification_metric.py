# This code is part of kartograf and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/cinnabar

from typing import Iterable
from scipy import stats
import numpy as np
import ast
import math


def _experiment_prediction_binning(experiment_dG: Iterable[float], predict_dG: Iterable[float], n_classes:int =2, best_class_fraction:float=None):
    """
    Helper function: bins the predicted and experimental values into n*n classes and gives back the number of occurance.

    """
    experiment_dG = np.array(experiment_dG)
    predict_dG = np.array(predict_dG)

    # Get clean bin Borders
    minV = np.min([experiment_dG, predict_dG])
    maxV = np.max([experiment_dG, predict_dG])

    if best_class_fraction is not None:
        step = (1-best_class_fraction) / (n_classes - 1)
        fracs = [ best_class_fraction ]
        for f in range(1, n_classes-1):
            fracs.append(f*step)
    else:
        step = 1 / n_classes
        fracs = [ best_class_fraction ]
        for f in range(1, n_classes-1):
            fracs.append(f*step)

    bin_borders = [minV]
    for f in fracs:
        upper_border = np.quantile(experiment_dG, f)
        bin_borders.append(upper_border)
    bin_borders.append(maxV)

    # categorization matrix
    binnings, borders, borders2 = np.histogram2d(experiment_dG, predict_dG,
                                                 bins=bin_borders)

    return binnings, borders


def _calculate_classification_accuracy(binnings, n_classes=2):
    """
    Helper function: calculates the accuracy of classification for a given 2D binned data set.
    """
    if(binnings.shape[0]!=n_classes):
        raise IOError(f"Clases not equal shape, {binnings.shape} vs {n_classes}")

    n_classifications = {}
    n_classifications[f"0Off"] = np.sum(np.diagonal(binnings))

    for n_off in range(1, n_classes):
        n_classifications[f"{n_off}Off"] = np.sum(np.diagonal(binnings, offset=n_off))
        n_classifications[f"{n_off}Off"] += np.sum(np.diagonal(binnings, offset=-n_off))

    weights = {'0Off': 1, '1Off': 0.5}
    weights.update({f"{nOff}Off": 1 for nOff in range(2, n_classes)})

    all_classifictions = np.sum([n_classifications[k] * weights[k] for k in n_classifications])+0.0001
    accuracy = n_classifications[f"0Off"] * weights[f"0Off"] / all_classifictions

    return accuracy, n_classifications


def classification_accuracy(experiment_dG:Iterable[float], perdict_dG: Iterable[float],
                            n_classes:int=2, n_resamples:int=300, best_class_fraction:float=None)->float:
    """
    Calculate the classification accuracy for two related experimental and predicted value vectors.
    The ordering of experiment and predicted needs to be identical!

    Parameters
    ----------
    experiment_dG : Iterable[float]
        iterable of experimental values, same sequence as predict_dG.
        (so experiment_dG[0] corresponds to predicted_dG[0], and so on)
    predict_dG: Iterable[float]
        iterable of predicted values, same sequence as experiment_dG.
    n_classes : int, optional
        number of classes, distributed across the datasets
    n_resamples: int, optional
        the number of metric recalculation for bootstrap error estimation
    best_class_fraction: float, optional
        the fraction of data that is part of the best class. (default: None => 1/n_classes)

    Returns
    -------
    float
        accuracy of the classification.

    """
    bins, borders = _experiment_prediction_binning(experiment_dG, perdict_dG, n_classes=n_classes, best_class_fraction=best_class_fraction)

    acc, n_classifications = _calculate_classification_accuracy(bins, n_classes=n_classes)

    def acc_boots_tfunc(data):
        d = np.array(list(map(ast.literal_eval, data)))
        x = d[:, 0]
        y = d[:, 1]
        bins, borders = _experiment_prediction_binning(x, y, n_classes=n_classes)
        acc, n_classifications = _calculate_classification_accuracy(bins, n_classes=n_classes)

        return acc

    data = list(map(str, zip(experiment_dG, perdict_dG)))
    s = stats.bootstrap([data], statistic=acc_boots_tfunc, n_resamples=n_resamples)

    return acc, s.standard_error


def _create_2d_histogram(y_true, y_pred):
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
    TypeError
        If `y_true` or `y_pred` cannot be converted to numpy arrays.
    """

    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
    except Exception as e:
        raise TypeError("Input data cannot be converted to numpy arrays.") from e

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    
    y_true_sorted = np.sort(y_true)
    y_pred_sorted = np.sort(y_pred)

    bins_true = np.concatenate(([y_true.min()], (y_true_sorted[:-1] + y_true_sorted[1:]) / 2, [y_true.max()]))
    bins_pred = np.concatenate(([y_pred.min()], (y_pred_sorted[:-1] + y_pred_sorted[1:]) / 2, [y_pred.max()]))

    histogram, bins_true, bins_pred = np.histogram2d(y_true, y_pred, bins=[bins_true, bins_pred])

    return histogram, bins_true, bins_pred


def _compute_overlap_coefficient(histogram, ranking):
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


def compute_fraction_best_ligands(y_true, y_pred, fraction=0.5):
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
    fraction : float, optional
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
    
    overlap_coefficients = [_compute_overlap_coefficient(histogram, i + 1) for i in range(num_ligands)]
    best_coefficients = overlap_coefficients[:num_best_ligands]
    
    fraction_best_ligands = sum(best_coefficients) / num_best_ligands

    return fraction_best_ligands