from typing import Literal, Union, get_args

import networkx as nx
import numpy as np
import scipy
import sklearn.metrics


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    r"""Compute root mean squared error between true and predicted values.

    Note
    ----
    The RMSE is calculated as:

    .. math:: RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}

    where :math:`y_i` is the predicted value and :math:`\hat{y}_i` is the true value.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    rmse : float
        RMSE between true and predicted values
    """
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


def calculate_mue(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    r"""Compute mean unsigned error between true and predicted values.

    Note
    ----
    The MUE is calculated as:

    .. math:: MUE = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|

    where :math:`y_i` is the predicted value and :math:`\hat{y}_i` is the true value.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    mue : float
        MUE between true and predicted values
    """
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)


def calculate_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    r"""Compute relative absolute error between true and predicted values.

    Note
    ----
    The RAE compares the mean absolute error of the predictions with a baseline model that always predicts the mean of the true values.
    It is calculated as:

    .. math:: RAE = \frac{\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|}{\frac{1}{N} \sum_{i=1}^N |\bar{y} - \hat{y}_i|}

    where :math:`y_i` is the predicted value, :math:`\hat{y}_i` is the true value, and :math:`\bar{y}` is the mean of the true values.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    rae : float
        RAE between true and predicted values
    """
    # mean unsigned error of the predictions
    mue = calculate_mue(y_true, y_pred)
    true_mean = np.mean(y_true)
    # mean absolute deviation of the true values from their mean
    mad = np.mean([np.abs(true_mean - i) for i in y_true])
    return mue / mad


def calculate_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute R^2 between true and predicted values.

    Note
    ----
    R^2 is calculated as the square of the Pearson correlation coefficient between true and predicted values.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    r2 : float
        R^2 between true and predicted values
    """
    r_value = calculate_pearson_r(y_true, y_pred)
    return r_value**2


def calculate_pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Pearson's r between true and predicted values.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    r : float
        Pearson's r between true and predicted values
    """
    return scipy.stats.pearsonr(y_true, y_pred)[0]


def calculate_kendalls_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Kendall's tau between true and predicted values.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    tau : float
        Kendall's tau between true and predicted values
    """
    return scipy.stats.kendalltau(y_true, y_pred)[0]


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""
    Compute the normalized root mean squared error between true and predicted values, using the true mean to normalize
    the RMSE. [1]_

    Note
    ----
    The NRMSE is calculated as:

    .. math:: NRMSE = \frac{RMSE}{\bar{y}}

    where :math:`RMSE` is the root mean squared error between true and predicted values, and :math:`\bar{y}` is the mean of the true values.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Returns
    -------
    nrmse : float
        NRMSE between true and predicted values

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root_mean_square_deviation
    """
    rmse = calculate_rmse(y_true, y_pred)
    mean_true = np.mean(y_true)
    return rmse / np.abs(mean_true)


def calculate_predictive_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Compute the predictive index as introduced by Pearlman et al. between true and predicted values. [1]_

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values

    Note
    ----
    The predictive index measures the correlation between the true and predicted values with a higher weight given to
    ligand pairs with larger true differences. The final value is between -1 and 1, where 1 indicates perfect ranking
    and -1 indicates perfectly anti-correlated ranking. It is calculated as:

    .. math:: PI = \frac{\sum^n_{j>i}\sum^n_{i}{W_{ij}C_{ij}}}{\sum^n_{j>i}\sum^n_{i}{W_{ij}}}

    where :math:`W_{ij} = abs(E_{j} - E_{i})` is a weight based on the true difference between the ligand pairs and :math:`C_{ij}`
    indicates if the rank ordering of the true differences agree with the predicted differences:

    .. math::

        C_{ij} = \begin{cases}
            1 & \text{if } (E_{j} - E_{i})/(P_{j} - P_{i}) > 0 \\
            0 & \text{if } (P_{j} - P_{i}) = 0 \\
            -1 & \text{if } (E_{j} - E_{i})/(P_{j} - P_{i}) < 0
        \end{cases}

    Returns
    -------
    pi : float
        Predictive index between true and predicted values between -1 and 1.

    References
    ----------
    .. [1] Pearlman, D.A. and Charifson, P.S., 2001. Are free energy calculations useful in practice? A comparison with rapid scoring functions for the p38 MAP kinase protein system. Journal of Medicinal Chemistry, 44(21), pp.3417-3423.
    """
    numerator, denominator = 0.0, 0.0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = np.abs(y_true[j] - y_true[i])
            # avoid division by zero when the predicted values are the same
            if y_pred[j] == y_pred[i]:
                c_ij = 0.0
            else:
                c_ij = np.sign((y_true[j] - y_true[i]) / (y_pred[j] - y_pred[i]))
            numerator += w_ij * c_ij
            denominator += w_ij
    return numerator / denominator


# map from statistic name to function that calculates the statistic
_AVAILABLE_STATS = {
    "RMSE": calculate_rmse,
    "NRMSE": calculate_nrmse,
    "MUE": calculate_mue,
    "RAE": calculate_rae,
    "R2": calculate_r2,
    "rho": calculate_pearson_r,
    "KTAU": calculate_kendalls_tau,
    "PI": calculate_predictive_index,
}
# make a type hint for the statistic names
Statistics = Literal["RMSE", "NRMSE", "MUE", "RAE", "R2", "rho", "KTAU", "PI"]
# make sure the type hint and the list stay in sync
assert set(get_args(Statistics)) == set(_AVAILABLE_STATS.keys())


def bootstrap_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dy_true: Union[np.ndarray, None] = None,
    dy_pred: Union[np.ndarray, None] = None,
    ci: float = 0.95,
    statistic: Statistics = "RMSE",
    nbootstrap: int = 1000,
    include_true_uncertainty: bool = False,
    include_pred_uncertainty: bool = False,
) -> dict:
    """Compute mean and confidence intervals of specified statistic.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values
    dy_true : ndarray with shape (N,) or None
        Errors of true values. If None, the values are assumed to have no errors
    dy_pred : ndarray with shape (N,) or None
        Errors of predicted values. If None, the values are assumed to have no errors
    ci : float, optional, default=0.95
        Interval for confidence interval (CI)
    statistic : str
        Statistic, one of ['RMSE', 'MUE', 'R2', 'rho', 'KTAU', 'RAE', 'NRMSE']
    nbootstrap : int, optional, default=1000
        Number of bootstrap samples
    include_true_uncertainty : bool, default False
        whether to account for the uncertainty in y_true when bootstrapping
    include_pred_uncertainty : bool, default False
        whether to account for the uncertainty in y_pred when bootstrapping

    Note
    -----
    If ``include_true_uncertainty`` or ``include_pred_uncertainty`` is True,
    normal noise will be added to the corresponding values during each bootstrap replicate.
    The standard deviation of the normal noise is taken from dy_true or dy_pred.

    Returns
    -------
    stats : dict of float
        'mle': statistic computed on the original data
        'mean' : mean value of the statistic over all bootstrap samples
        'stderr' : standard error of the statistic over all bootstrap samples
        'low' : low end of CI
        'high' : high end of CI
    """
    # check the statistic is valid
    if statistic not in _AVAILABLE_STATS:
        raise ValueError(f"unknown statistic {statistic}")
    stat_func = _AVAILABLE_STATS[statistic]

    if dy_true is None:
        dy_true = np.zeros_like(y_true)
    if dy_pred is None:
        dy_pred = np.zeros_like(y_pred)
    sample_size = len(y_true)
    # check the lengths of the inputs are the same and raise an error if not
    for arr in [y_pred, dy_true, dy_pred]:
        if len(arr) != sample_size:
            raise ValueError("All input arrays must have the same length")

    s_n = np.zeros([nbootstrap], np.float64)  # s_n[n] is the statistic computed for bootstrap sample n

    for replicate in range(nbootstrap):
        # draw bootstrap indices once and select values vectorized
        indices = np.random.choice(sample_size, size=sample_size, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        # only simulate normal noise when requested
        if include_true_uncertainty:
            std_true = np.fabs(dy_true[indices])
            y_true_sample = np.random.normal(loc=y_true_sample, scale=std_true)

        if include_pred_uncertainty:
            std_pred = np.fabs(dy_pred[indices])
            y_pred_sample = np.random.normal(loc=y_pred_sample, scale=std_pred)

        s_n[replicate] = stat_func(y_true_sample, y_pred_sample)

    # calculate the statistics and CI
    low_percentile = (1.0 - ci) / 2.0 * 100
    high_percentile = 100 - low_percentile
    stats = {
        "mle": stat_func(y_true, y_pred),  # the sample statistic
        "stderr": np.std(s_n),  # standard error of the bootstrap samples
        "mean": np.mean(s_n),  # mean of the bootstrap samples
        "low": np.percentile(s_n, low_percentile),  # low end of confidence interval
        "high": np.percentile(s_n, high_percentile),  # high end of confidence interval
    }
    return stats
