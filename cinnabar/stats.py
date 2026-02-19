from typing import Union

import networkx as nx
import numpy as np
import scipy
import sklearn.metrics

from cinnabar._due import Doi, due

due.cite(
    Doi("10.1021/acs.jcim.9b00528"),
    description="Compute maximum likelihood estimate of free energies and covariance in their estimates",
    path="cinnabar.stats.mle",
    cite_module=True,
)

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
    _, _, r_value, _, _ = scipy.stats.linregress(y_true, y_pred)
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
    the RMSE.

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
    """
    rmse = calculate_rmse(y_true, y_pred)
    mean_true = np.mean(y_true)
    return rmse / np.abs(mean_true)


AVAILABLE_STATS = {
    "RMSE": calculate_rmse,
    "NRMSE": calculate_nrmse,
    "MUE": calculate_mue,
    "RAE": calculate_rae,
    "R2": calculate_r2,
    "rho": calculate_pearson_r,
    "KTAU": calculate_kendalls_tau,
}

def bootstrap_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dy_true: Union[np.ndarray, None] = None,
    dy_pred: Union[np.ndarray, None] = None,
    ci: float = 0.95,
    statistic: str = "RMSE",
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
    if statistic not in AVAILABLE_STATS:
        raise ValueError(f'unknown statistic {statistic}')
    stat_func = AVAILABLE_STATS[statistic]

    # # not used?
    # def unique_differences(x):
    #     """Compute all unique differences"""
    #     N = len(x)
    #     return np.array([(x[i] - x[j]) for i in range(N) for j in range(N) if (i != j)])

    if dy_true is None:
        dy_true = np.zeros_like(y_true)
    if dy_pred is None:
        dy_pred = np.zeros_like(y_pred)
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(dy_true)
    assert len(y_true) == len(dy_pred)
    sample_size = len(y_true)
    s_n = np.zeros([nbootstrap], np.float64)  # s_n[n] is the statistic computed for bootstrap sample n

    for replicate in range(nbootstrap):
        # draw bootstrap indices once and select values vectorized
        indices = np.random.choice(sample_size, size=sample_size, replace=True)
        y_true_sample = y_true[indices].copy()
        y_pred_sample = y_pred[indices].copy()

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


def mle(graph: nx.DiGraph, factor: str = "f_ij", node_factor: Union[str, None] = None) -> np.ndarray:
    """
    Compute maximum likelihood estimate of free energies and covariance in their estimates.
    The number 'factor' is the node attribute on which the MLE will be calculated,
    where d'factor' will be used as the standard error of the factor

    Reference : https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528
    Xu, Huafeng. "Optimal measurement network of pairwise differences."
    Journal of Chemical Information and Modeling 59.11 (2019): 4720-4728.

    NOTE: Self-edges (edges that connect a node to itself) will be ignored.

    Parameters
    ----------
    graph :nx.Graph
        The graph for which an estimate is to be computed
        Each edge must have attributes 'f_ij' and 'df_ij' for the free energy and uncertainty
        estimate
        Will have 'bayesian_f_ij' and 'bayesian_df_ij' added to each edge
        and 'bayesian_f_i' and 'bayesian_df_i' added to each node.
    factor : string, default = 'f_ij'
        node attribute of nx.Graph that will be used for MLE
    node_factor : string, default = None
        optional - provide if there is node data (i.e. absolute values) 'f_i' or 'exp_DG' to
        include will expect a corresponding uncertainty 'f_di' or 'exp_dDG'
    Returns
    -------
    f_i : np.array with shape (n_ligands,)
        f_i[i] is the absolute free energy of ligand i in kcal/mol

    C : np.array with shape (n_ligands, n_ligands)
        C[i,j] is the covariance of the free energy estimates of i and j

    """
    # if we have bidirectional edge results we need to raise an error as they can not be used with MLE
    # track the edges we have seen
    edges = []
    for a, b in graph.edges:
        edge_name = tuple(sorted([a, b]))
        if edge_name in edges:
            raise ValueError(
                f"Multiple edges detected between nodes {a} and {b}. MLE cannot be performed on graphs with multiple "
                f"edges between the same nodes. The results should be combined into a single estimate and uncertainty "
                f"before performing MLE. See https://cinnabar.openfree.energy/en/latest/concepts/estimators.html#limitations for more details."
            )
        edges.append(edge_name)

    N = graph.number_of_nodes()
    if node_factor is None:
        f_ij = form_edge_matrix(graph, factor, action="antisymmetrize")
        df_ij = form_edge_matrix(graph, factor.replace("_", "_d"), action="symmetrize")
    else:
        f_ij = form_edge_matrix(graph, factor, action="antisymmetrize", node_label=node_factor)
        df_ij = form_edge_matrix(
            graph,
            factor.replace("_", "_d"),
            action="symmetrize",
            node_label=node_factor.replace("_", "_d"),
        )

    node_name_to_index = {}
    for i, name in enumerate(graph.nodes()):
        node_name_to_index[name] = i

    # Form F matrix (Eq 4)
    F_matrix = np.zeros([N, N])
    for a, b in graph.edges:
        i = node_name_to_index[a]
        j = node_name_to_index[b]
        if i == j:
            # The MLE solver will fail if we include self-edges, so we need to omit these
            continue
        F_matrix[i, j] = -(df_ij[i, j] ** (-2))
        F_matrix[j, i] = -(df_ij[i, j] ** (-2))
    for n in graph.nodes:
        i = node_name_to_index[n]
        if df_ij[i, i] == 0.0:
            F_matrix[i, i] = -np.sum(F_matrix[i, :])
        else:
            F_matrix[i, i] = df_ij[i, i] ** (-2) - np.sum(F_matrix[i, :])

    # Form z vector (Eq 3)
    z = np.zeros([N])
    for n in graph.nodes:
        i = node_name_to_index[n]
        if df_ij[i, i] != 0.0:
            z[i] = f_ij[i, i] * df_ij[i, i] ** (-2)
    for a, b in graph.edges:
        i = node_name_to_index[a]
        j = node_name_to_index[b]
        if i == j:
            # The MLE solver will fail if we include self-edges, so we need to omit these
            continue
        z[i] += f_ij[i, j] * df_ij[i, j] ** (-2)
        z[j] += f_ij[j, i] * df_ij[j, i] ** (-2)

    # Compute MLE estimate (Eq 2)
    Finv = np.linalg.pinv(F_matrix)
    f_i = np.matmul(Finv, z)

    # Compute uncertainty
    C = Finv
    return f_i, C


def form_edge_matrix(graph: nx.Graph, label: str, step=None, action=None, node_label=None) -> np.ndarray:
    """
    Extract the labeled property from edges into a matrix.

    Parameters
    ----------
    graph : nx.Graph
        The graph to extract data from
    label : str
        The label to use for extracting edge properties
    action : str, optional, default=None
        If 'symmetrize', returns a symmetric matrix A[i,j] = A[j,i]
        If 'antisymmetrize', returns an antisymmetric matrix A[i,j] = -A[j,i]
    node_label : sr, optional, default=None
        Diagonal will be occupied with absolute values, where labelled

    Returns
    ----------
    matrix
    """
    N = len(graph.nodes)
    matrix = np.zeros([N, N])

    node_name_to_index = {}
    for i, name in enumerate(graph.nodes()):
        node_name_to_index[name] = i

    for a, b in graph.edges:
        i = node_name_to_index[a]
        j = node_name_to_index[b]
        matrix[j, i] = graph.edges[a, b][label]
        if action == "symmetrize":
            matrix[i, j] = matrix[j, i]
        elif action == "antisymmetrize":
            matrix[i, j] = -matrix[j, i]
        elif action is None:
            pass
        else:
            # TODO use a more specific exception
            raise Exception(f'action "{action}" unknown.')

    if node_label is not None:
        for n in graph.nodes(data=True):
            i = node_name_to_index[n[0]]
            if node_label in n[1]:
                matrix[i, i] = n[1][node_label]

    return matrix
