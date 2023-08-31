import pathlib
from typing import Union

import openff.units
import pandas as pd
from openff.units import unit
import warnings
from typing import Optional, Hashable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import stats, ReferenceState, Measurement

_kcalpm = unit.kilocalorie_per_mole


def read_csv(filepath: pathlib.Path, units: Optional[openff.units.Quantity] = None) -> dict:
    """Read a legacy arsenic format csv file

    Parameters
    ----------
    filepath
      path to the csv file
    units : openff.units.Quantity, optional
      the units to use for values in the file, defaults to kcal/mol

    Returns
    -------
    raw_results : dict
      a dict with Experimental and Calculated keys
    """
    if units is None:
        warnings.warn("Assuming kcal/mol units on measurements")
        units = _kcalpm

    path_obj = pathlib.Path(filepath)
    raw_results = {"Experimental": {}, "Calculated": []}
    expt_block = False
    calc_block = False

    ground = ReferenceState()

    with path_obj.open() as f:
        for line in f:
            if "Experiment" in line:
                expt_block = True
            if "Calculate" in line or "Relative" in line:
                expt_block = False
                calc_block = True
            if expt_block and len(line.split(",")) == 3 and line[0] != "#":
                ligand, DG, dDG = line.split(",")
                expt = Measurement(labelA=ground,
                                   labelB=ligand,
                                   DG=float(DG) * units,
                                   uncertainty=float(dDG) * units,
                                   computational=False)
                raw_results["Experimental"][expt.labelB] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(',')

                calc = Measurement(labelA=ligA.strip(),
                                   labelB=ligB.strip(),
                                   DG=float(calc_DDG) * units,
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
    >>> g = ReferenceState()
    >>> experimental_result1 = Measurement(labelA=g, labelB="CAT-13a", DG=-8.83 * kJpm, uncertainty=0.10 * kJpm)
    >>> experimental_result2 = Measurement(labelA=g, labelB="CAT-17g", DG=-9.73 * kJpm, uncertainty=0.10 * kJpm)
    >>> # Load/create calculated results
    >>> calculated_result = Measurement(labelA="CAT-13a", labelB="CAT-17g", DG=0.36 * kJpm,
    ...                                 uncertainty=0.11 * kJpm)
    >>> # Incrementally created FEMap
    >>> fe = FEMap()
    >>> fe.add_measurement(experimental_result1)
    >>> fe.add_measurement(experimental_result2)
    >>> fe.add_measurement(calculated_result)
    """
    # graph with measurements as edges
    # absolute Measurements are an edge between 'ReferenceState' and the label
    # all edges are directed, all edges can be multiply defined
    graph: nx.MultiDiGraph

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    @classmethod
    def from_csv(cls, filename, units: Optional[unit.Quantity] = None):
        """Construct from legacy csv format"""
        data = read_csv(filename, units=units)

        # unpack data dictionary
        fe = cls()
        for r in data['Calculated']:
            fe.add_measurement(r)
        for r in data['Experimental'].values():
            fe.add_measurement(r)

        return fe

    def add_measurement(self, measurement: Measurement):
        """Add new observation to FEMap, modifies the FEMap in-place

        Any other attributes on the measurement are used as annotations

        Raises
        ------
        ValueError : if bad type given
        """
        # slurp out tasty data, anything but labels
        d = dict(measurement)
        d.pop('labelA', None)
        d.pop('labelB', None)

        # add both directions, but flip sign for the other direction
        d_backwards = {**d, 'DG': - d['DG'], 'source': 'reverse'}
        self.graph.add_edge(measurement.labelA, measurement.labelB, **d)
        self.graph.add_edge(measurement.labelB, measurement.labelA, **d_backwards)

    def add_experimental_measurement(self,
                                     label: str | Hashable,
                                     value: openff.units.Quantity,
                                     uncertainty: openff.units.Quantity,
                                     *,
                                     source: str = "",
                                     temperature=298.15 * unit.kelvin,
                                     ):
        """Add a single experimental measurement

        Parameters
        ----------
        label
          the ligand being measured
        value : openff.units.Quantity
          the measured value, as either Ki, IC50, kcal/mol, or kJ/mol.  The type
          of input is determined by the units of the input.
        uncertainty : openff.units.Quantity
          the uncertainty in the measurement
        source : str, optional
          an identifier for the source of the data
        temperature : openff.units.Quantity, optional
          the temperature the measurement was taken at, defaults to 298.15 K
        """
        if not isinstance(value, openff.units.Quantity):
            raise ValueError("Must include units with values, "
                             "e.g. openff.units.unit.kilocalorie_per_mole")

        if value.is_compatible_with('molar'):
            m = Measurement.from_experiment(label, value, uncertainty,
                                            source=source, temperature=temperature)
        else:  # value.is_compatible_with('kilocalorie_per_mole'):
            m = Measurement(
                labelA=ReferenceState(),
                labelB=label,
                DG=value,
                uncertainty=uncertainty,
                source=source, temperature=temperature,
                computational=False,
            )

        self.add_measurement(m)

    def add_relative_calculation(self,
                                 labelA: str | Hashable,
                                 labelB: str | Hashable,
                                 value: openff.units.Quantity,
                                 uncertainty: openff.units.Quantity,
                                 *,
                                 source: str = "",
                                 temperature=298.15 * unit.kelvin,
                                 ):
        """Add a single RBFE calculation

        Parameters
        ----------
        labelA, labelB
          the ligands being measured.  The measurement is taken from ligandA
          to ligandB, i.e. ligandA is the "old" or lambda=0.0 state, and ligandB
          is the "new" or lambda=1.0 state.
        value : openff.units.Quantity
          the measured DDG value, as kcal/mol, or kJ/mol.
        uncertainty : openff.units.Quantity
          the uncertainty in the measurement
        source : str, optional
          an identifier for the source of the data
        temperature : openff.units.Quantity, optional
          the temperature the measurement was taken at, defaults to 298.15 K
        """
        self.add_measurement(
            Measurement(
                labelA=labelA,
                labelB=labelB,
                DG=value,
                uncertainty=uncertainty,
                source=source,
                temperature=temperature,
                computational=True,
            )
        )

    def add_absolute_calculation(self,
                                 label,
                                 value: openff.units.Quantity,
                                 uncertainty: openff.units.Quantity,
                                 *,
                                 source: str = "",
                                 temperature=298.15 * unit.kelvin,
                                 ):
        """Add a single ABFE calculation

        Parameters
        ----------
        label
          the ligand being measured
        value : openff.units.Quantity
          the measured value, as kcal/mol, or kJ/mol.
        uncertainty : openff.units.Quantity
          the uncertainty in the measurement
        source : str, optional
          an identifier for the source of the data
        temperature : openff.units.Quantity, optional
          the temperature the measurement was taken at, defaults to 298.15 K
        """
        m = Measurement(
            labelA=ReferenceState(),
            labelB=label,
            DG=value, uncertainty=uncertainty,
            source=source, temperature=temperature,
            computational=True,
        )
        self.add_measurement(m)

    def get_relative_results(self) -> pd.DataFrame:
        """Gets a dataframe of all relative results

        The pandas DataFrame will have the following columns:
        - labelA
        - labelB
        - DDG
        - uncertainty
        - source
        - computational
        """
        kcpm = unit.kilocalorie_per_mole
        data = []
        for l1, l2, d in self.graph.edges(data=True):
            if d['source'] == 'reverse':
                continue
            if isinstance(l1, ReferenceState) or isinstance(l2, ReferenceState):
                continue

            data.append((
                l1, l2,
                d['DG'].to(kcpm).m, d['uncertainty'].to(kcpm).m,
                d['source'], d['computational']
            ))

        cols = [
            'labelA', 'labelB',
            'DDG (kcal/mol)', 'uncertainty (kcal/mol)',
            'source', 'computational'
        ]

        return pd.DataFrame(
            data=data,
            columns=cols,
        )

    def get_absolute_results(self) -> pd.DataFrame:
        """Get a dataframe of all absolute results

        The dataframe will have the following columns:
        - label
        - DG
        - uncertainty
        - source
        - computational
        """
        kcpm = unit.kilocalorie_per_mole
        data = []
        for l1, l2, d in self.graph.edges(data=True):
            if d['source'] == 'reverse':
                continue
            if not isinstance(l1, ReferenceState):
                continue
            if isinstance(l2, ReferenceState):
                continue

            data.append((
                l2,
                d['DG'].to(kcpm).m, d['uncertainty'].to(kcpm).m,
                d['source'], d['computational']
            ))

        cols = [
            'label',
            'DG (kcal/mol)', 'uncertainty (kcal/mol)',
            'source', 'computational'
        ]

        return pd.DataFrame(
            data=data,
            columns=cols,
        )

    @property
    def n_measurements(self) -> int:
        """Total number of both experimental and computational measurements"""
        return len(self.graph.edges) // 2

    @property
    def n_ligands(self) -> int:
        """Total number of unique ligands"""
        return len(self.ligands)

    @property
    def ligands(self) -> list:
        """All ligands in the graph"""
        # must ignore ReferenceState nodes
        return [n for n in self.graph.nodes
                if not isinstance(n, ReferenceState)]

    @property
    def degree(self) -> float:
        """Average degree of computational nodes"""
        return self.n_edges / self.n_ligands

    @property
    def n_edges(self) -> int:
        """Number of computational edges"""
        return sum(1 for _, _, d in self.graph.edges(data=True)
                   if d['computational']) // 2

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
        """Populate the FEMap with absolute computational values based on MLE"""
        # TODO: Make this return a new Graph with computational nodes annotated with DG values
        # TODO this could work if either relative or absolute expt values are provided
        mes = list(self.graph.edges(data=True))
        # for now, we must all be in the same units for this to work
        # grab unit of first measurement
        u = mes[0][-1]['DG'].u
        # check all over values are this unit
        if not all(d['DG'].u == u for _, _, d in mes):
            raise ValueError("All units must be the same")

        if self.check_weakly_connected():
            graph = self.to_legacy_graph()
            f_i_calc, C_calc = stats.mle(graph, factor="calc_DDG")
            variance = np.diagonal(C_calc) ** 0.5

            g = ReferenceState(label='MLE')

            for n, f_i, df_i in zip(graph.nodes, f_i_calc, variance):
                self.add_measurement(
                    Measurement(
                        labelA=g,
                        labelB=n,
                        DG=f_i * u,
                        uncertainty=df_i * u,
                        computational=True,
                        source='MLE',
                    )
                )

            # find all computational result labels
            comp_ligands = set()
            for A, B, d in self.graph.edges(data=True):
                if not d['computational']:
                    continue
                comp_ligands.add(A)
                comp_ligands.add(B)

            # find corresponding experimental results

            # use mean of experimental results to offset MLE reference point

            # add connection to MLE reference state and true reference state
            self.add_measurement(
                Measurement(
                    labelA=ReferenceState(),
                    labelB=g,
                    DG=0.1*u,
                    uncertainty=0.0 * u,
                    computational=True,
                    source='MLE',
                )
            )
        else:
            # TODO: This can eventually be worked around surely?
            raise ValueError("Computational results are not fully connected")

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
            if isinstance(a, ReferenceState):  # skip absolute measurements
                continue
            if d['source'] == 'reverse':  # skip mirrors
                continue

            g.add_edge(a, b, calc_DDG=d['DG'].magnitude, calc_dDDG=d['uncertainty'].magnitude)
        # add DG values from experiment graph
        for node, d in g.nodes(data=True):
            expt = self.graph.get_edge_data(ReferenceState(), node)
            if expt is None:
                continue
            expt = expt[0]

            d["exp_DG"] = expt['DG'].magnitude
            d["exp_dDG"] = expt['uncertainty'].magnitude
        # infer experiment DDG values
        for A, B, d in g.edges(data=True):
            try:
                DG_A = g.nodes[A]["exp_DG"]
                dDG_A = g.nodes[A]["exp_dDG"]
                DG_B = g.nodes[B]["exp_DG"]
                dDG_B = g.nodes[B]["exp_dDG"]
            except KeyError:
                continue
            else:
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
