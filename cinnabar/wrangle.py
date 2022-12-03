from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import stats


def read_csv(filename: str) -> dict:
    raw_results = {"Experimental": {}, "Calculated": []}
    expt_block = False
    calc_block = False
    with open(filename, "r") as f:
        for line in f:
            if "Experiment" in line:
                expt_block = True
            if "Calculate" in line or "Relative" in line:
                expt_block = False
                calc_block = True
            if expt_block and len(line.split(",")) == 3 and line[0] != "#":
                expt = ExperimentalResult(*line.split(","))
                raw_results["Experimental"][expt.ligand] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                calc = RelativeResult(*line.split(","))
                raw_results["Calculated"].append(calc)
    return raw_results


class RelativeResult(object):
    def __init__(self, ligandA, ligandB, calc_DDG, mbar_error, other_error):
        self.ligandA = str(ligandA).strip()
        self.ligandB = str(ligandB).strip()
        # scope for an experimental dDDG?
        self.calc_DDG = float(calc_DDG)
        self.mbar_dDDG = float(mbar_error)
        self.other_dDDG = float(other_error)

        # is this definitely always additive?
        self.calc_dDDG = self.mbar_dDDG + self.other_dDDG

    def toDF(self):
        # TODO - can we do the handling of the dataframe in a different way?
        # Or inside the plotting function that needs it?
        return pd.DataFrame(
            {
                "ligandA": self.ligandA,
                "ligandB": self.ligandB,
                "calc_DDG": self.calc_DDG,
                "mbar_dDDG": self.mbar_dDDG,
                "other_dDDG": self.other_dDDG,
                "calc_dDDG": self.calc_dDDG,
            },
            index=[f"{self.ligandA}_{self.ligandB}"],
        )


class ExperimentalResult(object):
    def __init__(self, ligand, expt_DG, expt_dDG):
        self.ligand = ligand
        self.DG = float(expt_DG)
        self.dDG = float(expt_dDG.strip("\n"))


class FEMap(object):
    def __init__(self, csv):
        self.results = read_csv(csv)
        self.graph = nx.DiGraph()

        self.generate_graph_from_results()

    def generate_graph_from_results(self):
        self._name_to_id = {}
        self._id_to_name = {}
        id = 0
        for result in self.results["Calculated"]:
            if result.ligandA not in self._name_to_id.keys():
                self._name_to_id[result.ligandA] = id
                self._id_to_name[id] = result.ligandA
                id += 1
            if result.ligandB not in self._name_to_id.keys():
                self._name_to_id[result.ligandB] = id
                self._id_to_name[id] = result.ligandB
                id += 1
            # # TODO need some exp error for mle to converge for exp... this is a horrible hack
            self.graph.add_edge(
                self._name_to_id[result.ligandA],
                self._name_to_id[result.ligandB],
                calc_DDG=result.calc_DDG,
                calc_dDDG=result.calc_dDDG,
            )

        for node in self.graph.nodes(data=True):
            name = self._id_to_name[node[0]]
            node[1]["name"] = name
            node[1]["exp_DG"] = self.results["Experimental"][name].DG
            node[1]["exp_dDG"] = self.results["Experimental"][name].dDG

        for edge in self.graph.edges(data=True):
            DG_A = self.graph.nodes[edge[0]]["exp_DG"]
            DG_B = self.graph.nodes[edge[1]]["exp_DG"]
            edge[2]["exp_DDG"] = DG_B - DG_A
            dDG_A = self.graph.nodes[edge[0]]["exp_dDG"]
            dDG_B = self.graph.nodes[edge[1]]["exp_dDG"]
            edge[2]["exp_dDDG"] = (dDG_A**2 + dDG_B**2) ** 0.5

        self.n_ligands = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        self.degree = self.n_edges / self.n_ligands

        # check the graph has minimal connectivity
        self.check_weakly_connected()
        if not self.weakly_connected:
            print("Graph is not connected enough to compute absolute values")
        else:
            self.generate_absolute_values()

    def check_weakly_connected(self):
        undirected_graph = self.graph.to_undirected()
        self.weakly_connected = nx.is_connected(undirected_graph)
        return nx.is_connected(undirected_graph)

    def generate_absolute_values(self):
        # TODO this could work if either relative or absolute expt values are provided
        if self.weakly_connected:
            f_i_calc, C_calc = stats.mle(self.graph, factor="calc_DDG")
            variance = np.diagonal(C_calc)
            for i, (f_i, df_i) in enumerate(zip(f_i_calc, variance**0.5)):
                self.graph.nodes[i]["calc_DG"] = f_i
                self.graph.nodes[i]["calc_dDG"] = df_i

    def draw_graph(self, title: str = "", filename: Union[str, None] = None):
        plt.figure(figsize=(10, 10))
        self._id_to_name = {}
        for i, j in self._name_to_id.items():
            self._id_to_name[j] = i
        nx.draw_circular(
            self.graph, labels=self._id_to_name, node_color="hotpink", node_size=250
        )
        long_title = f"{title} \n Nedges={self.n_edges} \n Nligands={self.n_ligands} \n Degree={self.degree:.2f}"
        plt.title(long_title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
