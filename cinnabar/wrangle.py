import pathlib
from typing import Union
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import stats


def read_csv(filepath: pathlib.Path) -> dict:
    path_obj = pathlib.Path(filepath)
    raw_results = {"Experimental": {}, "Calculated": []}
    expt_block = False
    calc_block = False
    with path_obj.open() as f:
        for line in f:
            if "Experiment" in line:
                expt_block = True
            if "Calculate" in line or "Relative" in line:
                expt_block = False
                calc_block = True
            if expt_block and len(line.split(",")) == 3 and line[0] != "#":
                ligand, DG, dDG = line.split(",")
                expt = ExperimentalResult(ligand, float(DG), float(dDG))
                raw_results["Experimental"][expt.ligand] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(',')

                calc = RelativeResult(ligA.strip(), ligB.strip(),
                                      float(calc_DDG), float(mbar_err),
                                      float(other_err))
                raw_results["Calculated"].append(calc)
    return raw_results


class RelativeResult:
    def __init__(self, ligandA: str, ligandB: str, calc_DDG: float,
                 mbar_error: float, other_error: float):
        self.ligandA = ligandA
        self.ligandB = ligandB
        # scope for an experimental dDDG?
        self.calc_DDG = calc_DDG
        self.mbar_dDDG = mbar_error
        self.other_dDDG = other_error

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


class ExperimentalResult:
    def __init__(self, ligand: str, expt_DG: float, expt_dDG: float):
        self.ligand = ligand
        self.DG = expt_DG
        self.dDG = expt_dDG


class FEMap:
    results: list[Union[RelativeResult, ExperimentalResult]]
    computational_graph: nx.MultiDiGraph
    experimental_graph: nx.MultiDiGraph

    _name_to_id: dict[str, int]
    _id_to_name: dict[int, str]

    def __init__(self, input_data: list[Union[RelativeResult, ExperimentalResult]]):
        """
        Construct Free Energy map of simulations from input data.

        Parameters
        ----------
            input_data: dict-like object
                dict-like object.

        Examples
        --------
        To read from a csv file specifically formatted for this, you can use:

        >>> fe = wrangle.FEMap.from_csv('../data/example.csv')

        To read from a dict-like object:

        >>> # Create experimental result
        >>> experimental_result1 = ExperimentalResult("CAT-13a", expt_DG=-8.83, expt_dDG=0.10)
        >>> experimental_result2 = ExperimentalResult("CAT-17g", expt_DG=-9.73, expt_dDG=0.10)
        >>> # Create calculated result
        >>> calculated_result = RelativeResult("CAT-13a", "CAT-17g", calc_DDG=0.36, mbar_error=0.11, other_error=0.0)
        >>> # Create object from dictionary
        >>> fe = FEMap([experimental_result1, experimental_result2, calculated_result])

        """
        self.results = input_data

        self.computational_graph = nx.MultiDiGraph()
        self.experimental_graph = nx.MultiDiGraph()

        self._name_to_id = {}
        self._id_to_name = {}
        idx = 0
        for result in self.results["Calculated"]:
            if result.ligandA not in self._name_to_id.keys():
                self._name_to_id[result.ligandA] = idx
                self._id_to_name[idx] = result.ligandA
                idx += 1
            if result.ligandB not in self._name_to_id.keys():
                self._name_to_id[result.ligandB] = idx
                self._id_to_name[idx] = result.ligandB
                idx += 1
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
            warnings.warn("Graph is not connected enough to compute absolute values")
        else:
            self.generate_absolute_values()

    @classmethod
    def from_csv(cls, filename):
        data = read_csv(filename)

        return cls(data)

    def check_weakly_connected(self):
        undirected_graph = self.graph.to_undirected()
        self.weakly_connected = nx.is_connected(undirected_graph)
        return nx.is_connected(undirected_graph)

    def generate_absolute_values(self):
        # TODO: Make this return a new Graph with computational nodes annotated with DG values

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
        nx.draw_circular(self.graph, labels=self._id_to_name, node_color="hotpink", node_size=250)
        long_title = f"{title} \n Nedges={self.n_edges} \n Nligands={self.n_ligands} \n Degree={self.degree:.2f}"
        plt.title(long_title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
