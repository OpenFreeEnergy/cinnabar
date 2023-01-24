import networkx as nx
import numpy as np
import scipy
import sklearn.metrics
from typing import Union


def bootstrap_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dy_true: Union[np.ndarray, None] = None,
    dy_pred: Union[np.ndarray, None] = None,
    ci: float = 0.95,
    statistic: str = "RMSE",
    nbootstrap: int = 1000,
    plot_type: str = "dG",
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
        Statistic, one of ['RMSE', 'MUE', 'R2', 'rho','KTAU','RAE']
    nbootstrap : int, optional, default=1000
        Number of bootstrap samples
    plot_type : str, optional, default='dG'
        'dG' or 'ddG'
    include_true_uncertainty : bool, default False
        whether to account for the uncertainty in y_true when bootstrapping
    include_pred_uncertainty : bool, default False
        whether to account for the uncertainty in y_pred when bootstrapping

    Returns
    -------
    rmse_stats : dict of float
        'mean' : mean RMSE
        'stderr' : standard error
        'low' : low end of CI
        'high' : high end of CI
    """

    def compute_statistic(y_true_sample: np.ndarray, y_pred_sample: np.ndarray, statistic: str):
        """Compute requested statistic.

        Parameters
        ----------
        y_true : ndarray with shape (N,)
            True values
        y_pred : ndarray with shape (N,)
            Predicted values
        statistic : str
            Statistic, one of ['RMSE', 'MUE', 'R2', 'rho','RAE','KTAU']

        """

        def calc_RAE(y_true_sample: np.ndarray, y_pred_sample: np.ndarray):
            MAE = sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
            mean = np.mean(y_true_sample)
            MAD = np.sum([np.abs(mean - i) for i in y_true_sample]) / float(len(y_true_sample))
            return MAE / MAD

        def calc_RRMSE(y_true_sample: np.ndarray, y_pred_sample: np.ndarray):
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample))
            mean_exp = np.mean(y_true_sample)
            mds = np.sum([(mean_exp - i) ** 2 for i in y_true_sample]) / float(len(y_true_sample))
            rrmse = np.sqrt(rmse**2 / mds)
            return rrmse

        if statistic == "RMSE":
            return np.sqrt(sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample))
        elif statistic == "MUE":
            return sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
        elif statistic == "R2":
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                y_true_sample, y_pred_sample
            )
            return r_value**2
        elif statistic == "rho":
            return scipy.stats.pearsonr(y_true_sample, y_pred_sample)[0]
        elif statistic == "RAE":
            return calc_RAE(y_true_sample, y_pred_sample)
        elif statistic == "KTAU":
            return scipy.stats.kendalltau(y_true_sample, y_pred_sample)[0]
        else:
            raise Exception("unknown statistic '{}'".format(statistic))

    # not used?
    def unique_differences(x):
        """Compute all unique differences"""
        N = len(x)
        return np.array([(x[i] - x[j]) for i in range(N) for j in range(N) if (i != j)])

    if dy_true is None:
        dy_true = np.zeros_like(y_true)
    if dy_pred is None:
        dy_pred = np.zeros_like(y_pred)
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(dy_true)
    assert len(y_true) == len(dy_pred)
    sample_size = len(y_true)
    s_n = np.zeros(
        [nbootstrap], np.float64
    )  # s_n[n] is the statistic computed for bootstrap sample n
    for replicate in range(nbootstrap):
        y_true_sample = np.zeros_like(y_true)
        y_pred_sample = np.zeros_like(y_pred)
        for i, j in enumerate(
            np.random.choice(np.arange(sample_size), size=[sample_size], replace=True)
        ):
            stddev_true = np.fabs(dy_true[j]) if include_true_uncertainty else 0
            stddev_pred = np.fabs(dy_pred[j]) if include_pred_uncertainty else 0
            y_true_sample[i] = np.random.normal(loc=y_true[j], scale=stddev_true, size=1)
            y_pred_sample[i] = np.random.normal(loc=y_pred[j], scale=stddev_pred, size=1)
        s_n[replicate] = compute_statistic(y_true_sample, y_pred_sample, statistic)

    rmse_stats = dict()
    rmse_stats["mle"] = compute_statistic(y_true, y_pred, statistic)
    rmse_stats["stderr"] = np.std(s_n)
    rmse_stats["mean"] = np.mean(s_n)
    # TODO: Is there a canned method to do this?
    s_n = np.sort(s_n)
    low_frac = (1.0 - ci) / 2.0
    high_frac = 1.0 - low_frac
    rmse_stats["low"] = s_n[int(np.floor(nbootstrap * low_frac))]
    rmse_stats["high"] = s_n[int(np.ceil(nbootstrap * high_frac))]

    return rmse_stats


def mle(
    graph: nx.DiGraph, factor: str = "f_ij", node_factor: Union[str, None] = None
) -> np.ndarray:
    """
    Compute maximum likelihood estimate of free energies and covariance in their estimates.
    The number 'factor' is the node attribute on which the MLE will be calculated,
    where d'factor' will be used as the standard error of the factor

    We assume the free energy of node 0 is zero.

    Reference : https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528
    Xu, Huafengraph. "Optimal measurement network of pairwise differences."
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
        f_i[i] is the absolute free energy of ligand i in kT
        f_i[0] = 0

    C : np.array with shape (n_ligands, n_ligands)
        C[i,j] is the covariance of the free energy estimates of i and j

    """
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
    for (a, b) in graph.edges:
        i = node_name_to_index[a]
        j = node_name_to_index[b]
        if i == j:
            # The MLE solver will fail if we include self-edges, so we need to omit these
            continue
        F_matrix[i, j] = -df_ij[i, j] ** (-2)
        F_matrix[j, i] = -df_ij[i, j] ** (-2)
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
    for (a, b) in graph.edges:
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


def form_edge_matrix(
    graph: nx.Graph, label: str, step=None, action=None, node_label=None
) -> np.ndarray:
    """
    Extract the labeled property from edges into a matrix
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
            raise Exception(f'action "{action}" unknown.')

    if node_label is not None:
        for n in graph.nodes(data=True):
            i = node_name_to_index[n[0]]
            if node_label in n[1]:
                matrix[i, i] = n[1][node_label]

    return matrix
