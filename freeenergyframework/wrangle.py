import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from freeenergyframework import stats
import pandas as pd


class RelativeResult(object):
    def __init__(self, ligandA, ligandB,
                 calc_DDG, mbar_error, other_error):
        self.ligandA = str(ligandA).strip()
        self.ligandB = str(ligandB).strip()
        # scope for an experimental dDDG?
        self.calc_DDG = float(calc_DDG)
        self.mbar_dDDG = float(mbar_error)
        self.other_dDDG = float(other_error)

        self.dcalc_DDG = self.mbar_dDDG+self.other_dDDG  # is this definitely always additive?
    def toDF(self):
      # TODO - can we do the handling of the dataframe in a different way? Or inside the plotting function that needs it?
        return pd.DataFrame({'ligandA': self.ligandA,
                             'ligandB': self.ligandB,
                             'calc_DDG': self.calc_DDG,
                             'mbar_dDDG': self.mbar_dDDG,
                             'other_dDDG': self.other_dDDG,
                             'dcalc_DDG': self.dcalc_DDG}, index=[f'{self.ligandA}_{self.ligandB}'])

class ExperimentalResult(object):
    def __init__(self, ligand, expt_DDG, expt_dDDG):
        self.ligand = ligand
        self.DDG = expt_DDG
        self.dDDG = expt_dDDG

class FEMap(object):

    def __init__(self, csv):
        self.graph = nx.DiGraph()
        self.n_edges = len(self.results)

        self.generate_graph_from_results()

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

        self.n_ligands = self.graph.number_of_nodes()
        self.degree = self.graph.number_of_edges() / self.n_ligands

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
            variance = np.diagonal(C_exp)
            for i, (f_i, df_i) in enumerate(zip(f_i_exp, variance**0.5)):
                self.graph.nodes[i]['f_i_exp'] = f_i
                self.graph.nodes[i]['df_i_exp'] = df_i

            f_i_calc, C_calc = stats.mle(self.graph, factor='calc_DDG')
            variance = np.diagonal(C_calc)
            for i, (f_i, df_i) in enumerate(zip(f_i_calc, variance**0.5)):
                self.graph.nodes[i]['f_i_calc'] = f_i
                self.graph.nodes[i]['df_i_calc'] = df_i

    def draw_graph(self, title='', filename=None):
        plt.figure(figsize=(10, 10))
        self._id_to_name = {}
        for i, j in self._name_to_id.items():
            self._id_to_name[j] = i
        nx.draw_circular(self.graph, labels=self._id_to_name, node_color='hotpink', node_size=250)
        long_title = f'{title} \n Nedges={self.n_edges} \n Nligands={self.n_ligands} \n Degree={self.degree:.2f}'
        plt.title(long_title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')


def read_csv(filename):
    raw_results = {'Experimental': [], 'Calculated': []}
    expt_block = False
    calc_block = False
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            if 'Experiment' in line:
                expt_block = True
                print('Doing expt block')
            if 'Calculate' in line or 'Relative' in line:
                expt_block = False
                calc_block = True
            print(expt_block)
            print(len(line.split(',')))
            if expt_block and len(line.split(',')) == 3 and line[0] != '#':
                expt = ExperimentalResult(*line.split(','))
                raw_results['Experimental'].append(expt)
            if calc_block and len(line.split(',')) == 5 and line[0] != '#':
                calc = RelativeResult(*line.split(','))
                raw_results['Calculated'].append(calc)
    return raw_results
