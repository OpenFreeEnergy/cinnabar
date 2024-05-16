# This code is part of kartograf and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/cinnabar

from typing import Iterable
from scipy import stats
import numpy as np
import ast


def _experiment_prediction_binning(experiment_dG: Iterable[float], predict_dG: Iterable[float], n_classes:int =2):
    """
    Helper function: bins the predicted and experimental values into n*n classes and gives back the number of occurance.

    """
    experiment_dG = np.array(experiment_dG)
    predict_dG = np.array(predict_dG)

    # Get clean bin Borders
    minV = np.min([experiment_dG, predict_dG])
    maxV = np.max([experiment_dG, predict_dG])

    step = 1 / n_classes
    bin_borders = [minV]
    for n in range(1, n_classes):
        upper_border = np.quantile(experiment_dG, n * step)
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
    bins, borders = _experiment_prediction_binning(experiment_dG, perdict_dG, n_classes=n_classes)

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


def FBVL(experiment_dG:Iterable[float], perdict_dG: Iterable[float], max_best_molecules_ratio:float=0.5)->float:
    """
    Metric inspired by the talk of Chris Bailey on alchemistry 2024
    """
    raise NotImplementedError()

    fbvl_score = 0

    return fbvl_score