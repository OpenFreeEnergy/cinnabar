from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from openff.arsenic import stats


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


def read_perses_results(input_data, ligand_id_to_name, exp_data, exp_error):
    """
    Read FE calculation results from a list of perses Simulation objects.

    Parameters
    ----------
        input_data: list
            List of perses.analysis.load_simulations.Simulation objects to extract the data from.
        ligand_id_to_name: dict
            Mapping of ligand id (index) to ligand name.
        exp_data: list
            List with the corresponding experimental values in kcal/mol in the same order as input data.
        exp_error: list
            List of uncertainties in experimental values, in kcal/mol.
    """
    # Perses results require a ligand id to name map
    if ligand_id_to_name is None:
        raise ValueError("Expected ligand id to name map. Received None.")

    raw_results = {"Experimental": {}, "Calculated": []}

    # loop through data and fill dictionary
    for sim in input_data:
        # extracting indices for ligands
        ligA = int(sim.directory[3:].split('_')[1])  # define edges
        ligB = int(sim.directory[3:].split('_')[2])
        calc_DDG=-sim.bindingdg / sim.bindingdg.unit
        calc_DDG_dev=sim.bindingddg / sim.bindingddg.unit
        # Create Relative object from data
        liga_name = ligand_id_to_name[ligA]
        ligb_name = ligand_id_to_name[ligB]
        raw_results['Calculated'].append(RelativeResult(liga_name, ligb_name, calc_DDG, calc_DDG_dev, calc_DDG_dev))
        # Create Experimental objects from data and add to dictionary
        # TODO: Redundant. This overwrites already set values (not blocking)
        liganda_name = ligand_id_to_name[ligA]
        ligandb_name = ligand_id_to_name[ligA]
        raw_results['Experimental'][liganda_name] = ExperimentalResult(liganda_name, exp_data[ligA], exp_error[ligA])
        raw_results['Experimental'][ligandb_name] = ExperimentalResult(ligandb_name, exp_data[ligA], exp_error[ligA])

    return raw_results


class RelativeResult(object):
    def __init__(self, ligandA, ligandB, calc_DDG, mbar_error, other_error):
        self.ligandA = str(ligandA).strip()
        self.ligandB = str(ligandB).strip()
        # scope for an experimental dDDG?
        self.calc_DDG = float(calc_DDG)
        self.mbar_dDDG = float(mbar_error)
        self.other_dDDG = float(other_error)

        self.calc_dDDG = self.mbar_dDDG + self.other_dDDG  # is this definitely always additive?

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
        try:
            self.dDG = float(expt_dDG.strip("\n"))
        except AttributeError:
            # Data is already numeric
            self.dDG = float(expt_dDG)


class BaseFEMap(object):
    """Creates a graph-based map of a free energy set of calculations."""
    def __init__(self):
        """
        Construct Free Energy map of simulations from input data.

        Parameters
        ----------
            input_data: csv file path or list(perses.analysis.load_simulations.Simulation)
                File path to csv file or instance of Simulation.
            ligand_id_to_name: dict, optional
                Dictionary with the ligand id to name mapping to be used. Defaults to None. Reading from perses data
                requires to specify a ligand_id_to_name.
        """
        self.graph = nx.DiGraph()
        self._name_to_id = {}
        self._id_to_name = {}
        self.n_ligands = self.graph.number_of_nodes()

    def check_weakly_connected(self):
        undirected_graph = self.graph.to_undirected()
        self.weakly_connected = nx.is_connected(undirected_graph)
        return nx.is_connected(undirected_graph)

    def generate_absolute_values(self):
        # TODO this could work if either relative or absolute expt values are provided
        if self.weakly_connected:
            f_i_calc, C_calc = stats.mle(self.graph, factor="calc_DDG")
            variance = np.diagonal(C_calc)
            for i, (f_i, df_i) in enumerate(zip(f_i_calc, variance ** 0.5)):
                self.graph.nodes[i]["calc_DG"] = f_i
                self.graph.nodes[i]["calc_dDG"] = df_i


class FEMap(BaseFEMap):
    """Creates a graph-based map of a free energy set of calculations."""
    def __init__(self, input_data, ligand_id_to_name=None, experimental_data=None, experimental_error=None):
        """
        Construct Free Energy map of simulations from input data.

        Parameters
        ----------
            input_data: csv file path or list(perses.analysis.load_simulations.Simulation)
                File path to csv file or instance of Simulation.
            ligand_id_to_name: dict, optional
                Dictionary with the ligand id to name mapping to be used. Defaults to None. If not specified a map
                will be build, arbitrarily.
        """
        # Call constructor of parent class
        super().__init__()
        # Read results depending on input data format
        if isinstance(input_data, str):
            self.results = read_csv(input_data)
        else:
            self.results = read_perses_results(input_data,
                                               ligand_id_to_name,
                                               experimental_data,
                                               experimental_error)
        self.n_edges = len(self.results)
        self.generate_graph_from_results()
        self.degree = self.graph.number_of_edges() / self.n_ligands

    def generate_graph_from_results(self):
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
            edge[2]["exp_dDDG"] = (dDG_A ** 2 + dDG_B ** 2) ** 0.5

        self.n_ligands = self.graph.number_of_nodes()
        self.degree = self.graph.number_of_edges() / self.n_ligands

        # check the graph has minimal connectivity
        self.check_weakly_connected()
        if not self.weakly_connected:
            print("Graph is not connected enough to compute absolute values")
        else:
            self.generate_absolute_values()

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
