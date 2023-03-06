import pathlib
from typing import Union
from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import stats


def read_csv(filepath: pathlib.Path) -> dict:
    warnings.warn("Assuming kJ/mol units on measurements")

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
                expt = AbsoluteMeasurement(ligand=ligand,
                                           DG=float(DG),
                                           uncertainty=float(dDG),
                                           computational=False)
                raw_results["Experimental"][expt.ligand] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(',')

                calc = RelativeMeasurement(ligandA=ligA.strip(),
                                           ligandB=ligB.strip(),
                                           DDG=float(calc_DDG),
                                           uncertainty=float(mbar_err) + float(other_err),
                                           computational=True)
                raw_results["Calculated"].append(calc)
    return raw_results


class RelativeMeasurement(DefaultModel):
    ligandA: str
    ligandB: str
    DDG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool


class AbsoluteMeasurement(DefaultModel):
    ligand: str
    DG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool


class FEMap:
    """Free Energy map of both simulations and bench measurements

    Examples
    --------
    To read from a csv file specifically formatted for this, you can use:

    >>> fe = wrangle.FEMap.from_csv('../data/example.csv')

    To read from a dict-like object:

    >>> # Load/create experimental results
    >>> experimental_result1 = ExperimentalResult("CAT-13a", expt_DG=-8.83, expt_dDG=0.10)
    >>> experimental_result2 = ExperimentalResult("CAT-17g", expt_DG=-9.73, expt_dDG=0.10)
    >>> # Load/create calculated results
    >>> calculated_result = RelativeMeasurement("CAT-13a", "CAT-17g", calc_DDG=0.36, mbar_error=0.11, other_error=0.0)
    >>> # Incrementally created FEMap
    >>> fe = FEMap()
    >>> fe.add_measurement(experimental_result1)
    >>> fe.add_measurement(experimental_result2)
    >>> fe.add_measurement(calculated_result)
    """
    # MultiGraph allows multiple *relative* measurements, but not multiple identical nodes
    # so maybe instead make absolute results all relative to a dummy node
    computational_graph: nx.MultiDiGraph
    experimental_graph: nx.MultiDiGraph

    @classmethod
    def from_csv(cls, filename):
        """Construct from legacy csv format"""
        data = read_csv(filename)

        # unpack data dictionary
        fe = cls()
        for r in data['Calculated']:
            fe.add_measurement(r)
        for r in data['Experimental'].values():
            fe.add_measurement(r)

        return fe

    def add_measurement(self, measurement: Union[RelativeMeasurement, AbsoluteMeasurement]):
        """Add new observation to FEMap, modifies in place

        Any other attributes on the measurement are used as annotations

        Raises
        ------
        ValueError : if bad type given
        """
        if isinstance(measurement, AbsoluteMeasurement):
            # coerse to relative to simplify logic
            meas_ = RelativeMeasurement(ligandA='NULL',
                                        ligandB=measurement.ligand,
                                        DDG=measurement.DG,
                                        uncertainty=measurement.uncertainty,
                                        computational=measurement.computational)
        elif isinstance(measurement, RelativeMeasurement):
            meas_ = measurement
        else:
            raise TypeError()

        # slurp out tasty data, anything but labels
        d = dict(meas_)
        d.pop('ligandA', None)
        d.pop('ligandB', None)
        d.pop('ligand', None)

        if meas_.computational:
            self.computational_graph.add_edge(meas_.ligandA, meas_.ligandB, **d)
        else:
            self.experimental_graph.add_edge(meas_.ligandA, meas_.ligandB, **d)

    @property
    def n_measurements(self) -> int:
        """Total number of both experimental and computational measurements"""
        return len(self.experimental_graph.edges) + len(self.computational_graph.edges)

    @property
    def n_ligands(self) -> int:
        """Total number of unique ligands"""
        # must ignore NULL sentinel node
        exptl = self.experimental_graph.nodes - {'NULL'}
        compt = self.computational_graph.nodes

        return len(exptl | compt)

    @property
    def degree(self) -> float:
        """Average degree of all nodes"""
        return self.n_measurements / self.n_ligands

    def is_weakly_connected(self) -> bool:
        # todo; cache
        undirected_graph = self.computational_graph.to_undirected()
        return nx.is_connected(undirected_graph)

    def generate_absolute_values(self):
        # TODO: Make this return a new Graph with computational nodes annotated with DG values

        # TODO this could work if either relative or absolute expt values are provided
        if self.is_weakly_connected():
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
