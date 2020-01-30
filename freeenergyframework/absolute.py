def generate_absolute_values(results):
    import networkx as nx
    from freeenergyframework import stats
    import numpy as np
    name_to_id = {}
    id = 0
    G = nx.DiGraph()
    for result in results:
        if result.ligandA not in name_to_id.keys():
            name_to_id[result.ligandA] = id
            id += 1
        if result.ligandB not in name_to_id.keys():
            name_to_id[result.ligandB] = id
            id += 1
            # TODO need some exp error for mle to converge for exp... this is a horrible hack
        if result.dexp_DDG == 0.0:
            result.dexp_DDG = 0.01
        G.add_edge(name_to_id[result.ligandA], name_to_id[result.ligandB], exp_DDG=result.exp_DDG, dexp_DDG=result.dexp_DDG,
                   calc_DDG=result.calc_DDG, dcalc_DDG=result.dcalc_DDG)

    f_i_exp, C_exp = stats.mle(G, factor='exp_DDG')
    f_i_calc, C_calc = stats.mle(G, factor='calc_DDG')
    return f_i_exp, f_i_calc, np.diagonal(C_exp), np.diagonal(C_calc)
