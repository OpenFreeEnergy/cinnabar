import pathlib
from typing import Union

import openff.units
from openff.units import unit
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import stats
from .measurements import RelativeMeasurement, AbsoluteMeasurement

_kJpm = unit.kilojoule_per_mole


def read_csv(filepath: pathlib.Path, units: Optional[openff.units.Quantity] = None) -> dict:
    if units is None:
        warnings.warn("Assuming kJ/mol units on measurements")
        units = _kJpm

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
                                           DG=float(DG) * units,
                                           uncertainty=float(dDG) * units,
                                           computational=False)
                raw_results["Experimental"][expt.label] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(',')

                calc = RelativeMeasurement(labelA=ligA.strip(),
                                           labelB=ligB.strip(),
                                           DDG=float(calc_DDG) * units,
                                           uncertainty=(float(mbar_err) + float(other_err)) * units,
                                           computational=True)
                raw_results["Calculated"].append(calc)
    return raw_results


class FEMap:
    """Free Energy map of both simulations and bench measurements

    Examples
    --------
    To read from a csv file specifically formatted for this, you can use:

    >>> fe = FEMap.from_csv('../data/example.csv')

    To construct manually:

    >>> # Load/create experimental results
    >>> from openff.units import unit
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
    # graph with measurements as edges
    # absolute Measurements are an edge between 'NULL' and the label
    # all edges are directed, all edges can be multiply defined
    graph: nx.MultiDiGraph

    def __init__(self):
        self.graph = nx.MultiDiGraph()

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

        self.graph.add_edge(meas_.labelA, meas_.labelB, **d)

    @property
    def n_measurements(self) -> int:
        """Total number of both experimental and computational measurements"""
        return len(self.graph.edges)

    @property
    def n_ligands(self) -> int:
        """Total number of unique ligands"""
        # must ignore NULL sentinel node
        return len(self.graph.nodes - {'NULL'})

    @property
    def degree(self) -> float:
        """Average degree of computational nodes"""
        return self.n_edges / self.n_ligands

    @property
    def n_edges(self) -> int:
        """Number of computational edges"""
        return sum(1 for _, _, d in self.graph.edges(data=True) if d['computational'])

    def check_weakly_connected(self) -> bool:
        """Checks if all results in the graph are reachable from other results"""
        # todo; cache
        comp_graph = nx.MultiGraph()
        for a, b, d in self.graph.edges(data=True):
            if not d['computational']:
                continue
            comp_graph.add_edge(a, b)

        return nx.is_connected(comp_graph)

    def generate_absolute_values(self):
        # TODO: Make this return a new Graph with computational nodes annotated with DG values

        # TODO this could work if either relative or absolute expt values are provided
        if self.check_weakly_connected():
            graph = self.to_legacy_graph()

            f_i_calc, C_calc = stats.mle(graph, factor="calc_DDG")
            variance = np.diagonal(C_calc)
            for i, (f_i, df_i) in enumerate(zip(f_i_calc, variance**0.5)):
                graph.nodes[i]["calc_DG"] = f_i
                graph.nodes[i]["calc_dDG"] = df_i

    def to_legacy_graph(self) -> nx.DiGraph:
        """Produce single graph version of this FEMap

        This graph will feature:
        - experimental DDG values calculated as the difference between experimental DG values
        - calculated DG values calculated via mle

        This matches the legacy format of this object, notably:
        - drops multi edge capability
        - removes units from values
        """
        # reduces to nx.DiGraph
        g = nx.DiGraph()
        # add DDG values from computational graph
        for a, b, d in self.graph.edges(data=True):
            if not d['computational']:
                continue
            g.add_edge(a, b, calc_DDG=d['DDG'].magnitude, calc_dDDG=d['uncertainty'].magnitude)
        # add DG values from experiment graph
        for node, d in g.nodes(data=True):
            expt = self.graph.get_edge_data('NULL', node)[0]

            d["exp_DG"] = expt['DDG'].magnitude
            d["exp_dDG"] = expt['uncertainty'].magnitude
        # infer experiment DDG values
        for A, B, d in g.edges(data=True):
            DG_A = g.nodes[A]["exp_DG"]
            dDG_A = g.nodes[A]["exp_dDG"]
            DG_B = g.nodes[B]["exp_DG"]
            dDG_B = g.nodes[B]["exp_dDG"]

            d["exp_DDG"] = DG_B - DG_A
            d["exp_dDDG"] = (dDG_A**2 + dDG_B**2) ** 0.5
        # apply MLE for calculated DG values
        if self.check_weakly_connected():
            f_i_calc, C_calc = stats.mle(g, factor="calc_DDG")
            variance = np.diagonal(C_calc)
            variance = variance ** 0.5

            for (_, d), f_i, df_i in zip(g.nodes(data=True), f_i_calc, variance):
                d['calc_DG'] = f_i
                d['calc_dDG'] = df_i
        else:
            warnings.warn("Graph is not connected enough to compute absolute values")

        return g

    def draw_graph(self, title: str = "", filename: Union[str, None] = None):
        plt.figure(figsize=(10, 10))

        graph = self.to_legacy_graph()

        labels = {n: n for n in graph.nodes}

        nx.draw_circular(graph, labels=labels, node_color="hotpink", node_size=250)
        long_title = f"{title} \n Nedges={self.n_edges} \n Nligands={self.n_ligands} \n Degree={self.degree:.2f}"
        plt.title(long_title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
