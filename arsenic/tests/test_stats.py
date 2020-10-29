def test_mle_easy(input_absolutes=[-14., -13., -9.]):
    """Test that the MLE for a graph with an absolute
    estimate on all nodes will recapitulate it

    """
    import networkx as nx
    import numpy as np
    from arsenic import stats

    # test data
    input_absolutes = [-14., -13., -9.]

    g = nx.DiGraph()
    for i, f in enumerate(input_absolutes):
        g.add_node(i, f_i=f, f_di=0.5)

    edges = [(0, 1), (0, 2), (2, 1)]
    for a, b in edges:
        noise = np.random.uniform(low=-1., high=1.)
        diff = input_absolutes[b] - input_absolutes[a] + noise
        g.add_edge(a, b, f_ij=diff, f_dij=0.5+np.abs(noise))

    output_absolutes, C = stats.mle(g, factor='f_ij', node_factor='f_i')

    for i, n in enumerate(g.nodes(data=True)):
        diff = np.abs(output_absolutes[i] - input_absolutes[i])
        assert diff < C[i, i], f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."


def test_mle_hard(input_absolutes=[-14., -13., -9.]):
    """Test that the MLE for a graph with a node missing an absolute value can get it right based on relative results

    """
    import networkx as nx
    import numpy as np
    from arsenic import stats
    input_absolutes = [-14., -13., -9.]
    # make a t
    g = nx.DiGraph()
    # Don't assign the first absolute value, check that MLE can get close to it
    for i, f in enumerate(input_absolutes):
        if i == 0:
            g.add_node(i)
        else:
            g.add_node(i, f_i=f, f_di=0.5)

    edges = [(0, 1), (0, 2), (2, 1)]
    for a, b in edges:
        noise = np.random.uniform(low=-1., high=1.)
        diff = input_absolutes[b] - input_absolutes[a] + noise
        g.add_edge(a, b, f_ij=diff, f_dij=0.5+np.abs(noise))

    output_absolutes, C = stats.mle(g, factor='f_ij', node_factor='f_i')

    for i, n in enumerate(g.nodes(data=True)):
        diff = np.abs(output_absolutes[i] - input_absolutes[i])
        assert diff < C[i, i], f"MLE error. Output absolute \
         estimate, {output_absolutes[i]}, is too far from\
         true value: {input_absolutes[i]}."


def test_mle_relative(input_absolutes=[-14., -13., -9.]):
    """Test that the MLE can get the relative differences correct when no absolute values are provided

    """
    import networkx as nx
    import numpy as np
    from arsenic import stats
    import itertools

    g = nx.DiGraph()
    # Don't assign any absolute values
    edges = [(0, 1), (0, 2), (2, 1)]
    for a, b in edges:
        noise = np.random.uniform(low=-0.5, high=0.5)
        diff = input_absolutes[b] - input_absolutes[a] + noise
        g.add_edge(a, b, f_ij=diff, f_dij=0.5+np.abs(noise))

    output_absolutes, C = stats.mle(g, factor='f_ij', node_factor='f_i')

    pairs = itertools.combinations(range(len(input_absolutes)), 2)

    for i, j in pairs:
        mle_diff = output_absolutes[i] - output_absolutes[j]
        true_diff = input_absolutes[i] - input_absolutes[j]

        assert np.abs(true_diff - mle_diff) < 1., f"Relative\
         difference from MLE: {mle_diff} is too far from the\
         input difference, {true_diff}"


def test_correlation_positive():
    """ Test that the absolute DG plots have the correct signs, and statistics within reasonable agreement to the example data in `arsenic/data/example.csv`

    """
    from arsenic import plotting, stats, wrangle
    import os
    print(os.system('pwd'))
    import numpy as np
    fe = wrangle.FEMap('arsenic/data/example.csv')
    
    x_data = np.asarray([node[1]['exp_DG'] for node in fe.graph.nodes(data=True)])
    y_data = np.asarray([node[1]['calc_DG'] for node in fe.graph.nodes(data=True)])
    xerr = np.asarray([node[1]['exp_dDG'] for node in fe.graph.nodes(data=True)])
    yerr = np.asarray([node[1]['calc_dDG'] for node in fe.graph.nodes(data=True)])
    
    s = stats.bootstrap_statistic(x_data, y_data, xerr, yerr, statistic='rho')
    assert 0 < s['mle'] < 1, 'Correlation must be positive for this data'
    
    for stat in ['RMSE','MUE','R2','rho']:
        s = stats.bootstrap_statistic(x_data, y_data, xerr, yerr, statistic=stat)
        # all of the statistics for this example is between 0.61 and 0.84
        assert 0.5 < s['mle'] < 0.9, f"Correlation must be positive for this data. {stat} is {s['mle']}"
