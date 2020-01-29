import numpy as np

def bootstrap_statistic(y_true, y_pred, ci=0.95, statistic='RMSE', nbootstrap = 1000, plot_type='dG'):
    import sklearn.metrics
    import scipy.stats
    """Compute mean and confidence intervals of specified statistic.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values
    y_pred : ndarray with shape (N,)
        Predicted values
    ci : float, optional, default=0.95
        Interval for CI
    statistic : str
        Statistic, one of ['RMSE', 'MUE', 'R2', 'rho']
    nbootstrap : int, optional, default=1000
        Number of bootstrap samples
    plot_type : str, optional, default='dG'
        'dG' or 'ddG'

    Returns
    -------
    rmse_stats : dict of floeat
        'mean' : mean RMSE
        'stderr' : standard error
        'low' : low end of CI
        'high' : high end of CI
    """

    def compute_statistic(y_true_sample, y_pred_sample, statistic):
        """Compute requested statistic.

        Parameters
        ----------
        y_true : ndarray with shape (N,)
            True values
        y_pred : ndarray with shape (N,)
            Predicted values
        statistic : str
            Statistic, one of ['RMSE', 'MUE', 'R2', 'rho']

        """
        if statistic == 'RMSE':
            return np.sqrt(sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample))
        elif statistic == 'MUE':
            return sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
        elif statistic == 'R2':
#             return sklearn.metrics.r2_score(y_true_sample, y_pred_sample)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true_sample, y_pred_sample)
            return r_value**2
        elif statistic == 'rho':
            return scipy.stats.pearsonr(y_true_sample, y_pred_sample)[0]
        else:
            raise Exception("unknown statistic '{}'".format(statistic))

    def unique_differences(x):
        """Compute all unique differences"""
        N = len(x)
        return np.array( [ (x[i] - x[j]) for i in range(N) for j in range(N) if (i != j) ] )

    assert len(y_true) == len(y_pred)
    sample_size = len(y_true)
    s_n = np.zeros([nbootstrap], np.float64) # s_n[n] is the statistic computed for bootstrap sample n
    for replicate in range(nbootstrap):
        indices = np.random.choice(np.arange(sample_size), size=[sample_size])
        if plot_type == 'dG':
            y_true_sample, y_pred_sample = y_true[indices], y_pred[indices]
        elif plot_type == 'ddG':
            y_true_sample, y_pred_sample = unique_differences(y_true[indices]), unique_differences(y_pred[indices])
        s_n[replicate] = compute_statistic(y_true_sample, y_pred_sample, statistic)

    rmse_stats = dict()
    if plot_type == 'dG':
        rmse_stats['mle'] = compute_statistic(y_true, y_pred, statistic)
    elif plot_type == 'ddG':
        rmse_stats['mle'] = compute_statistic(unique_differences(y_true), unique_differences(y_pred), statistic)
    rmse_stats['stderr'] = np.std(s_n)
    rmse_stats['mean'] = np.mean(s_n)
    # TODO: Is there a canned method to do this?
    s_n = np.sort(s_n)
    low_frac = (1.0-ci)/2.0
    high_frac = 1.0 - low_frac
    rmse_stats['low'] = s_n[int(np.floor(nbootstrap*low_frac))]
    rmse_stats['high'] = s_n[int(np.ceil(nbootstrap*high_frac))]

    return rmse_stats

def mle(g,factor='f_ij'):
    """
    Compute maximum likelihood estimate of free energies and covariance in their estimates.

    We assume the free energy of node 0 is zero.

    Parameters
    ----------
    g : nx.Graph
        The graph for which an estimate is to be computed
        Each edge must have attributes 'f_ij' and 'df_ij' for the free energy and uncertainty estimate
        Will have 'bayesian_f_ij' and 'bayesian_df_ij' added to each edge
        and 'bayesian_f_i' and 'bayesian_df_i' added to each node.

    Returns
    -------
    f_i : np.array with shape (n_ligands,)
        f_i[i] is the absolute free energy of ligand i in kT
        f_i[0] = 0

    C : np.array with shape (n_ligands, n_ligands)
        C[i,j] is the covariance of the free energy estimates of i and j

    """
    N = len(g.nodes)
    f_ij = form_edge_matrix(g, factor, action='antisymmetrize')
    df_ij = form_edge_matrix(g, f'd{factor}', action='symmetrize')

    # Form F matrix (Eq 4)
    F = np.zeros([N,N])
    for (i,j) in g.edges:
        F[i,j] = - df_ij[i,j]**(-2)
        F[j,i] = - df_ij[i,j]**(-2)
    for i in g.nodes:
        F[i,i] = - np.sum(F[i,:])

    # Form z vector (Eq 3)
    z = np.zeros([N])
    for (i,j) in g.edges:
        z[i] += f_ij[i,j] * df_ij[i,j]**(-2)
        z[j] += f_ij[j,i] * df_ij[j,i]**(-2)

    # Compute MLE estimate (Eq 2)
    Finv = np.linalg.pinv(F)
    f_i = - np.matmul(Finv, z) # NOTE: This differs in sign from Eq. 2!
    f_i[:] -= f_i[0]

    # Compute uncertainty
    C = Finv
    return f_i, C


def form_edge_matrix(g, label, step=None, action=None):
    """
    Extract the labeled property from edges into a matrix

    Parameters
    ----------
    g : nx.Graph
        The graph to extract data from
    label : str
        The label to use for extracting edge properties
    action : str, optional, default=None
        If 'symmetrize', will return a symmetric matrix where A[i,j] = A[j,i]
        If 'antisymmetrize', will return an antisymmetric matrix where A[i,j] = -A[j,i]

    """
    N = len(g.nodes)
    matrix = np.zeros([N,N])
    for i, j in g.edges:
        matrix[i,j] = g.edges[i,j][label]
        if action == 'symmetrize':
            matrix[j,i] = matrix[i,j]
        elif action == 'antisymmetrize':
            matrix[j,i] = -matrix[i,j]
        elif action is None:
            pass
        else:
            raise Exception(f'action "{action}" unknown.')
    return matrix
