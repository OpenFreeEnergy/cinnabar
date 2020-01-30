import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from freeenergyframework import stats

class Result(object):
    def __init__(self, ligandA, ligandB,
                 exp_DDG, exp_dDDG,
                 calc_DDG, mbar_error, other_error):
        self.ligandA = str(ligandA)
        self.ligandB = str(ligandB)
        self.exp_DDG = float(exp_DDG)
        self.dexp_DDG = float(exp_dDDG)
        # scope for an experimental dDDG?
        self.calc_DDG = float(calc_DDG)
        self.mbar_dDDG = float(mbar_error)
        self.other_dDDG = float(other_error)
        self.dcalc_DDG = self.mbar_dDDG+self.other_dDDG  # is this definitely always additive?


class FEMap(object):

    def __init__(self, results):
        self.results = results
        self.graph = nx.DiGraph()

        self.generate_graph_from_results()

        # check the graph has minimal connectivity

    def generate_graph_from_results(self):
        self._name_to_id = {}
        id = 0
        for result in self.results:
            if result.ligandA not in self._name_to_id.keys():
                self._name_to_id[result.ligandA] = id
                id += 1
            if result.ligandB not in self._name_to_id.keys():
                self._name_to_id[result.ligandB] = id
                id += 1
            # TODO need some exp error for mle to converge for exp... this is a horrible hack
            if result.dexp_DDG == 0.0:
                result.dexp_DDG = 0.01
            self.graph.add_edge(self._name_to_id[result.ligandA], self._name_to_id[result.ligandB],
            exp_DDG=result.exp_DDG, dexp_DDG=result.dexp_DDG,
            calc_DDG=result.calc_DDG, dcalc_DDG=result.dcalc_DDG)


        # check the graph has minimal connectivity
        self.check_weakly_connected()
        if not self.weakly_connected:
            print('Graph is not connected enough to compute absolute values')
        else:
            self.generate_absolute_values()

    def check_weakly_connected(self):
        undirected_graph = self.graph.to_undirected()
        self.weakly_connected = nx.is_connected(undirected_graph)
        return nx.is_connected(undirected_graph)

    def generate_absolute_values(self):
        if self.weakly_connected:
            f_i_exp, C_exp = stats.mle(self.graph, factor='exp_DDG')
            for i, (f_i, df_i) in enumerate(zip(f_i_exp, np.diagonal(C_exp))):
                self.graph.node[i]['f_i_exp'] = f_i
                self.graph.node[i]['df_i_exp'] = df_i

            f_i_calc, C_calc = stats.mle(self.graph, factor='calc_DDG')
            for i, (f_i, df_i) in enumerate(zip(f_i_calc, np.diagonal(C_calc))):
                self.graph.node[i]['f_i_calc'] = f_i
                self.graph.node[i]['df_i_calc'] = df_i

    def draw_graph(self, title='', filename=None):
        plt.figure(figsize=(10, 10))
        self._id_to_name = {}
        for i, j in self._name_to_id.items():
            self._id_to_name[j] = i
        nx.draw_circular(self.graph, labels=self._id_to_name, node_color='hotpink', node_size=250)
        plt.title(title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)


def read_csv(filename):
    raw_results = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] != '#':
                raw_results.append(Result(*line.split(',')))
    return raw_results
