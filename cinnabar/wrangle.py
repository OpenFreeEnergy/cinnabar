import pathlib
from typing import Union
from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import stats


_kJpm = unit.kilojoule_per_mole


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
                expt = AbsoluteMeasurement(label=ligand,
                                           DG=float(DG) * _kJpm,
                                           uncertainty=float(dDG) * _kJpm,
                                           computational=False)
                raw_results["Experimental"][expt.label] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(',')

                calc = RelativeMeasurement(labelA=ligA.strip(),
                                           labelB=ligB.strip(),
                                           DDG=float(calc_DDG) * _kJpm,
                                           uncertainty=(float(mbar_err) + float(other_err)) * _kJpm,
                                           computational=True)
                raw_results["Calculated"].append(calc)
    return raw_results


class RelativeMeasurement(DefaultModel):
    """The free energy difference of moving from A to B"""
    labelA: str
    labelB: str
    DDG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool


class AbsoluteMeasurement(DefaultModel):
    label: str
    DG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool


class FEMap:
    """Free Energy map of both simulations and bench measurements

    Examples
    --------
    To read from a csv file specifically formatted for this, you can use:

    >>> fe = wrangle.FEMap.from_csv('../data/example.csv')

    To construct manually:

    >>> # Load/create experimental results
    >>> kJpm = unit.kilojoule_per_mole
    >>> experimental_result1 = AbsoluteMeasurement(label="CAT-13a", DG=-8.83 * kJpm, uncertainty=0.10 * kJpm)
    >>> experimental_result2 = AbsoluteMeasurement(label="CAT-17g", DG=-9.73 * kJpm, uncertainty=0.10 * kJpm)
    >>> # Load/create calculated results
    >>> calculated_result = RelativeMeasurement(labelA="CAT-13a", labelB="CAT-17g", DDG=0.36 * kJpm,
    ...                                         uncertainty=0.11 * kJpm)
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

    def __init__(self):
        self.computational_graph = nx.MultiDiGraph()
        self.experimental_graph = nx.MultiDiGraph()

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
        """Add new observation to FEMap, modifies the FEMap in-place

        Any other attributes on the measurement are used as annotations

        Raises
        ------
        ValueError : if bad type given
        """
        if isinstance(measurement, AbsoluteMeasurement):
            # coerce to relative to simplify logic
            meas_ = RelativeMeasurement(labelA='NULL',
                                        labelB=measurement.label,
                                        DDG=measurement.DG,
                                        uncertainty=measurement.uncertainty,
                                        computational=measurement.computational)
        elif isinstance(measurement, RelativeMeasurement):
            meas_ = measurement
        else:
            raise ValueError()

        # slurp out tasty data, anything but labels
        d = dict(meas_)
        d.pop('labelA', None)
        d.pop('labelB', None)
        d.pop('label', None)

        if meas_.computational:
            self.computational_graph.add_edge(meas_.labelA, meas_.labelB, **d)
        else:
            self.experimental_graph.add_edge(meas_.labelA, meas_.labelB, **d)

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
