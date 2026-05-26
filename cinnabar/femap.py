"""
FEMap
=====

The workhorse of cinnabar, a :class:`FEMap` contains many measurements of free energy differences,
both relative and absolute,
which form an interconnected "network" of values.
"""

import copy
import math
import itertools
import pathlib
import warnings
from dataclasses import asdict
from typing import TYPE_CHECKING, Hashable, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openff.units
import pandas as pd
from openff.units import Quantity, unit

from cinnabar import stats
from cinnabar.measurements import Measurement, ReferenceState

if TYPE_CHECKING:
    from cinnabar.estimators import Estimator, EstimatorResult  # pragma: no cover

_kcalpm = unit.kilocalorie_per_mole


def read_csv(filepath: pathlib.Path, units: Optional[openff.units.Quantity] = None) -> dict:
    """Read a legacy format csv file

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
                expt = Measurement(
                    labelA=ground,
                    labelB=ligand,
                    DG=float(DG) * units,
                    uncertainty=float(dDG) * units,
                    computational=False,
                )
                raw_results["Experimental"][expt.labelB] = expt
            if calc_block and len(line.split(",")) == 5 and line[0] != "#":
                ligA, ligB, calc_DDG, mbar_err, other_err = line.split(",")

                calc = Measurement(
                    labelA=ligA.strip(),
                    labelB=ligB.strip(),
                    DG=float(calc_DDG) * units,
                    uncertainty=(float(mbar_err) + float(other_err)) * units,
                    computational=True,
                )
                raw_results["Calculated"].append(calc)
    return raw_results


class FEMap:
    """Free Energy map of both simulations and bench measurements

    Contains a set (non-duplicate entries) of different measurements.

    Examples
    --------
    To construct a FEMap by hand:

    >>> # Load/create experimental results
    >>> from openff.units import unit
    >>> kJpm = unit.kilojoule_per_mole
    >>> g = ReferenceState()
    >>> experimental_result1 = Measurement(labelA=g, labelB="CAT-13a", DG=-8.83 * kJpm, uncertainty=0.10 * kJpm,
    ...                                    computational=False)
    >>> experimental_result2 = Measurement(labelA=g, labelB="CAT-17g", DG=-9.73 * kJpm, uncertainty=0.10 * kJpm,
    ...                                    computational=False)
    >>> # Load/create calculated results
    >>> calculated_result = Measurement(labelA="CAT-13a", labelB="CAT-17g", DG=0.36 * kJpm,
    ...                                 uncertainty=0.11 * kJpm, computational=True)
    >>> # Incrementally created FEMap
    >>> fe = FEMap()
    >>> fe.add_measurement(experimental_result1)
    >>> fe.add_measurement(experimental_result2)
    >>> fe.add_measurement(calculated_result)

    To read from a legacy csv file specifically formatted for this, you can use:

    >>> fe = FEMap.from_csv('../data/example.csv')
    """

    # internal representation:
    # graph with measurements as edges
    # absolute Measurements are an edge between 'ReferenceState' and the label
    # all edges are directed
    # all edges can be multiply defined
    _graph: nx.MultiDiGraph
    _estimator_metadata: dict[str, "EstimatorResult"]

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._estimator_metadata = {}

    def __iter__(self):
        for a, b, d in self._graph.edges(data=True):
            # skip artificial reverse edges
            if d["source"] == "reverse":
                continue

            yield Measurement(labelA=a, labelB=b, **d)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        # iter returns hashable Measurements, so this will compare contents
        return set(self) == set(other)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # deduplicate based on hashing the Measurements from iter
        my_items = set(self)
        other_items = set(other)

        new = self.__class__()
        for m in my_items | other_items:
            new.add_measurement(m)

        return new

    def __len__(self):
        return len(list(iter(self)))

    def to_networkx(self) -> nx.MultiDiGraph:
        """A *copy* of the FEMap as a networkx Graph

        The FEMap is represented as a multi-edged directional graph

        Edges have the following attributes:

        * DG: the free energy difference of going from the first edge label to
          the second edge label
        * uncertainty: uncertainty of the DG value
        * temperature: the temperature at which DG was measured
        * computational: boolean label of the original source of the data
        * source: a string describing the source of data.

        Note
        ----
        All edges appear twice, once with the attribute source='reverse',
        and the DG value flipped.  This allows "pathfinding" like approaches,
        where the DG values will be correctly summed.
        """
        return copy.deepcopy(self._graph)

    @classmethod
    def from_networkx(cls, graph: nx.MultiDiGraph):
        """Create FEMap from network representation

        Note
        ----
        Currently absolutely no validation of the input is done.
        """
        m = cls()
        m._graph = graph

        return m

    @classmethod
    def from_csv(cls, filename, units: Optional[Quantity] = None):
        """Construct from legacy csv format"""
        data = read_csv(filename, units=units)

        # unpack data dictionary
        fe = cls()
        for r in data["Calculated"]:
            fe.add_measurement(r)
        for r in data["Experimental"].values():
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
        d = asdict(measurement)
        d.pop("labelA", None)
        d.pop("labelB", None)

        # add both directions, but flip sign for the other direction
        d_backwards = {**d, "DG": -d["DG"], "source": "reverse"}
        self._graph.add_edge(measurement.labelA, measurement.labelB, **d)
        self._graph.add_edge(measurement.labelB, measurement.labelA, **d_backwards)

    def add_experimental_measurement(
        self,
        label: Union[str, Hashable],
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
            raise ValueError("Must include units with values, e.g. openff.units.unit.kilocalorie_per_mole")

        if value.is_compatible_with("molar"):
            m = Measurement.from_experiment(label, value, uncertainty, source=source, temperature=temperature)
        else:  # value.is_compatible_with('kilocalorie_per_mole'):
            m = Measurement(
                labelA=ReferenceState(),
                labelB=label,
                DG=value,
                uncertainty=uncertainty,
                source=source,
                temperature=temperature,
                computational=False,
            )

        self.add_measurement(m)

    def add_relative_calculation(
        self,
        labelA: Union[str, Hashable],
        labelB: Union[str, Hashable],
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

    def add_absolute_calculation(
        self,
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
            DG=value,
            uncertainty=uncertainty,
            source=source,
            temperature=temperature,
            computational=True,
        )
        self.add_measurement(m)

    def get_relative_dataframe(self) -> pd.DataFrame:
        """Get a dataframe of all relative results for all sources including experimental and computational.

        Note
        ----
        The pandas DataFrame will have the following columns:
        - labelA
        - labelB
        - DDG (kcal/mol)
        - uncertainty (kcal/mol)
        - source
        - computational
        Only simulated relative results are included for the computational results
        The dataframe will be sorted by source, computational, labelA, and labelB to ensure consistent ordering of results between sources.
        """
        kcpm = unit.kilocalorie_per_mole
        data = []
        # store the non-computational results so we can include them in the dataframe for the same edges
        non_comp_results = {}
        for l1, l2, d in self._graph.edges(data=True):
            if d["source"] == "reverse":
                continue
            if isinstance(l1, ReferenceState) or isinstance(l2, ReferenceState) and not d["computational"]:
                label = l2 if isinstance(l1, ReferenceState) else l1
                non_comp_results[label] = d
                continue

            data.append((l1, l2, d["DG"].to(kcpm).m, d["uncertainty"].to(kcpm).m, d["source"], d["computational"]))

        # for each computational result add the experimental result for the same edge if it exists
        comp_data = []
        for l1, l2, *_ in data:
            exp_1 = non_comp_results.get(l1, None)
            exp_2 = non_comp_results.get(l2, None)
            if exp_1 is not None and exp_2 is not None:
                # if we have both, we can add the experimental DDG and uncertainty to the dataframe
                exp_ddg = exp_2["DG"].to(kcpm).m - exp_1["DG"].to(kcpm).m
                exp_uncertainty = (exp_1["uncertainty"].to(kcpm).m ** 2 + exp_2["uncertainty"].to(kcpm).m ** 2) ** 0.5
                comp_data.append((l1, l2, exp_ddg, exp_uncertainty, "experimental", False))

        cols = ["labelA", "labelB", "DDG (kcal/mol)", "uncertainty (kcal/mol)", "source", "computational"]

        df = pd.DataFrame(
            data=data + comp_data,
            columns=cols,
        )
        return df.sort_values(by=["source", "computational", "labelA", "labelB"]).reset_index(drop=True)

    def get_absolute_dataframe(self) -> pd.DataFrame:
        """Get a dataframe of all absolute results from all sources.

        Note
        ----
        The dataframe will have the following columns:
        - label
        - DG (kcal/mol)
        - uncertainty (kcal/mol)
        - source
        - computational
        The dataframe will be sorted by source, computational, and label to ensure consistent ordering of results between sources.
        """
        kcpm = unit.kilocalorie_per_mole
        data = []
        for l1, l2, d in self._graph.edges(data=True):
            if d["source"] == "reverse":
                continue
            if not isinstance(l1, ReferenceState):
                continue
            if isinstance(l2, ReferenceState):
                continue

            data.append((l2, d["DG"].to(kcpm).m, d["uncertainty"].to(kcpm).m, d["source"], d["computational"]))

        cols = ["label", "DG (kcal/mol)", "uncertainty (kcal/mol)", "source", "computational"]

        df = pd.DataFrame(
            data=data,
            columns=cols,
        )
        return df.sort_values(by=["source", "computational", "label"]).reset_index(drop=True)

    def get_all_to_all_relative_dataframe(self, symmetrical: bool = True) -> pd.DataFrame:
        """Get a dataframe of the all-to-all pairwise relative results using the absolute DG values.

        Parameters
        ----------
        symmetrical : bool, optional
            If True, include both directions of each pairwise comparison. If False, include only one direction (default is True).

        Returns
        -------
        df : pd.DataFrame
             A dataframe containing all pairwise relative results.

        Note
        ----
        The dataframe will have the following columns:
        - labelA
        - labelB
        - DDG (kcal/mol)
        - uncertainty (kcal/mol)
        - source
        - computational
        The dataframe will be sorted by source, computational, labelA, and labelB to ensure that pairing order is consistent.
        If `symmetrical` is True, the dataframe will include both (labelA, labelB) and (labelB, labelA) for each pair of labels, with opposite signs for DDG and the same uncertainty.
        If an estimator is used to generate the absolute binding affinities from relative results this function attempts
        to use the covariance_matrix in the uncertainty if available, if not the covariance is set to zero.
        """
        # we need to group by the source and computational labels and then compute the pairwise differences within each group, then concatenate the results together
        df = self.get_absolute_dataframe()
        grouped = df.groupby(["source", "computational"])
        pairwise_dfs = []
        for (source, computational), group in grouped:
            # sort the group by label to ensure that pairing order is consistent
            group = group.sort_values(by="label")
            labels = group["label"].values
            dgs = group["DG (kcal/mol)"].values
            uncertainties = group["uncertainty (kcal/mol)"].values
            if computational:
                # get the estimator metadata as we need the covariance matrix, but it may not be
                # available (e.g. ABFE-only data added directly without running an estimator)
                try:
                    estimator_metadata = self.get_estimator_metadata(source)
                except KeyError:
                    estimator_metadata = None
            else:
                estimator_metadata = None

            data = []
            for i, j in itertools.combinations(range(len(labels)), 2):
                label_a, label_b = labels[i], labels[j]
                # transformation i -> j has a DDG of j - i
                ddg = dgs[j] - dgs[i]
                # get the covariance if metadata with a covariance matrix is available,
                # otherwise assume zero covariance (e.g. ABFE-only data, or a custom estimator
                # that does not expose a covariance matrix) this will slightly degrade the error estimate in some cases
                covariance = 0.0
                if (
                    estimator_metadata is not None
                    and hasattr(estimator_metadata, "covariance_matrix")
                    and hasattr(estimator_metadata, "ligand_order")
                ):
                    try:
                        ligand_i = estimator_metadata.ligand_order.index(label_a)
                        ligand_j = estimator_metadata.ligand_order.index(label_b)
                        covariance = estimator_metadata.covariance_matrix[ligand_i, ligand_j]
                    except ValueError:
                        # label not found in ligand_order — fall back to zero covariance
                        covariance = 0.0
                uncertainty = (uncertainties[i] ** 2 + uncertainties[j] ** 2 - 2 * covariance) ** 0.5
                data.append((label_a, label_b, ddg, uncertainty, source, computational))
                if symmetrical:
                    data.append((label_b, label_a, -ddg, uncertainty, source, computational))

            pairwise_df = pd.DataFrame(
                data=data,
                columns=["labelA", "labelB", "DDG (kcal/mol)", "uncertainty (kcal/mol)", "source", "computational"],
            )
            pairwise_dfs.append(pairwise_df)

        if not pairwise_dfs:
            return pd.DataFrame(
                columns=["labelA", "labelB", "DDG (kcal/mol)", "uncertainty (kcal/mol)", "source", "computational"]
            )
        return (
            pd.concat(pairwise_dfs)
            .sort_values(by=["source", "computational", "labelA", "labelB"])
            .reset_index(drop=True)
        )

    @property
    def n_measurements(self) -> int:
        """Total number of both experimental and computational measurements"""
        return len(self._graph.edges) // 2

    @property
    def n_ligands(self) -> int:
        """Total number of unique ligands"""
        return len(self.ligands)

    @property
    def ligands(self) -> list:
        """All ligands in the graph"""
        # must ignore ReferenceState nodes
        return [n for n in self._graph.nodes if not isinstance(n, ReferenceState)]

    @property
    def degree(self) -> float:
        """Average degree of computational nodes"""
        return self.n_edges / self.n_ligands

    @property
    def n_edges(self) -> int:
        """Number of computational edges"""
        return sum(1 for _, _, d in self._graph.edges(data=True) if d["computational"]) // 2

    def check_weakly_connected(self) -> bool:
        """
        Checks if all computational results in the graph are reachable from other results.

        Returns
        -------
        bool
             True if the graph is weakly connected, False otherwise.

        Raises
        ------
        ValueError
            If the graph contains no computational edges.
        """
        # todo; cache
        comp_graph = nx.MultiGraph()
        for a, b, d in self._graph.edges(data=True):
            if not d["computational"]:
                continue
            comp_graph.add_edge(a, b)

        try:
            is_connected = nx.is_connected(comp_graph)
            return is_connected
        except nx.NetworkXPointlessConcept:
            raise ValueError("Graph contains no computational edges, cannot check connectivity")

    def generate_absolute_values(self, estimator: Optional["Estimator"] = None):
        """Populate the FEMap with absolute computational values.

        Runs the estimator on this femap for each unique computational
        source, adds the returned ``Measurement`` objects, and stores the
        ``EstimatorResult`` metadata per source for later retrieval via
        ``get_estimator_metadata``.

        Parameters
        ----------
        estimator : Estimator, optional
            The estimator to use.  Defaults to
            the MLEEstimator.

        Raises
        ------
        ValueError
            If measurements have mixed units or the computational graph for
            any source is not weakly connected.

        See Also
        --------
        get_estimator_metadata : retrieve stored metadata after estimation.

        Notes
        -----
        * This method modifies the FEMap in-place, adding new measurements and metadata.
        * The estimator is run separately for each unique computational source, predictions will have a new source tag of
            the form ``{estimator_name}({original_source})``, e.g. ``MLE(openff-2.0.0)``.

        """
        mes = list(self._graph.edges(data=True))
        if not mes:
            raise ValueError("FEMap contains no measurements")
        u = mes[0][-1]["DG"].u
        if not all(d["DG"].u == u for _, _, d in mes):
            raise ValueError("All units must be the same")

        if estimator is None:
            from cinnabar.estimators import MLEEstimator

            estimator = MLEEstimator()

        # estimate() returns {composed_source: (measurements, result)},
        # where composed_source is e.g. "MLE" or "MLE(openff-2.0.0)" depending on the number of input sources.
        # the same keys are used in _estimator_metadata so that
        # get_estimator_metadata can retrieve the result for a given source.
        results_by_source = estimator.estimate(self)
        for composed_source, (measurements, result) in results_by_source.items():
            for m in measurements:
                self.add_measurement(m)
            self._estimator_metadata[composed_source] = result

    def get_estimator_metadata(self, source: str) -> "EstimatorResult":
        """Retrieve stored metadata from a previous :meth:`generate_absolute_values` call.

        Parameters
        ----------
        source : str
            The composed source identifier for the estimator results to retrieve, e.g. ``MLE(openff-2.0.0)``.

        Returns
        -------
        EstimatorResult
            The concrete type depends on the estimator used, e.g.
            :class:`~cinnabar.estimators.MLEEstimatorResult` for
            :class:`~cinnabar.estimators.MLEEstimator`.

        Raises
        ------
        KeyError
            If no metadata is stored for the provided source.
        """
        if source not in self._estimator_metadata:
            available = list(self._estimator_metadata.keys())
            raise KeyError(
                f"No estimator metadata stored for source {source}. "
                f"Available sources: {available}. "
                "Call generate_absolute_values() first."
            )
        return self._estimator_metadata[source]

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
        # the MLE method can only use a single result per edge, we need to raise and error if we have repeats or bidirectional results
        edges_seen = []
        # add DDG values from computational graph
        for a, b, d in self._graph.edges(data=True):
            if not d["computational"]:
                continue
            if isinstance(a, ReferenceState):  # skip absolute measurements
                continue
            if d["source"] == "reverse":  # skip mirrors
                continue
            edge_name = tuple(sorted([a, b]))
            if edge_name in edges_seen:
                raise ValueError(
                    f"Multiple edges detected between nodes {a} and {b}. MLE cannot be performed on graphs with multiple "
                    f"edges between the same nodes. The results should be combined into a single estimate and uncertainty "
                    f"before performing MLE. See https://cinnabar.openfree.energy/en/latest/concepts/estimators.html#limitations for more details."
                )

            g.add_edge(a, b, calc_DDG=d["DG"].magnitude, calc_dDDG=d["uncertainty"].magnitude)
            edges_seen.append(edge_name)
        # add DG values from experiment graph
        for node, d in g.nodes(data=True):
            expt = self._graph.get_edge_data(ReferenceState(), node)
            if expt is None:
                continue
            expt = expt[0]

            d["exp_DG"] = expt["DG"].magnitude
            d["exp_dDG"] = expt["uncertainty"].magnitude
            # name of the node used to add data labels to plots
            d["name"] = node
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
            variance = variance**0.5

            for (_, d), f_i, df_i in zip(g.nodes(data=True), f_i_calc, variance):
                d["calc_DG"] = f_i
                d["calc_dDG"] = df_i
        else:
            warnings.warn("Graph is not connected enough to compute absolute values")

        return g

    def draw_graph(self, title: str = "", filename: Union[str, None] = None):
        """
        Draw the graph using matplotlib.

        Parameters
        ----------
        title : str, optional
            Title for the graph.
        filename : str or None, optional
            If provided, the graph will be saved to this file. If None, the graph will be displayed.
        """
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

    def get_cycle_closure(self, max_cycle_length: int = 5) -> list[tuple[str, float]]:
        """
        Calculate cycle closure errors for all cycles in the network.

        Parameters
        ----------
        max_cycle_length : int, optional
            Only consider cycles up to this length. Default 5.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'cycle', 'cc (kcal/mol)', sorted by
            cycle closure error descending.
        """
        network = self.to_legacy_graph()
        edge_ddg = {(a, b): d["calc_DDG"] for a, b, d in network.edges(data=True)}
        edge_uncertainty = {(a, b): d["calc_dDDG"] for a, b, d in network.edges(data=True)}

        # Find all ligand cycles
        cycles = [c for c in nx.simple_cycles(network.to_undirected()) if len(c) <= max_cycle_length]

        # Loop over cycles, calculate sum of DG along cycle
        rows = []
        for cycle in cycles:
            # Store DDG values along the cycle
            sum_ddgs = 0.0
            sum_var = 0.0
            for inx, ligand in enumerate(cycle):
                lig_a = ligand
                lig_b = cycle[inx + 1] if inx < len(cycle) - 1 else cycle[0]

                # depending on the direction the edge was calculated,
                # the sign of the DDG has to change
                if (lig_a, lig_b) in edge_ddg:
                    sum_ddgs += edge_ddg[(lig_a, lig_b)]
                    sum_var += edge_uncertainty[(lig_a, lig_b)] ** 2
                elif (lig_b, lig_a) in edge_ddg:
                    sum_ddgs -= edge_ddg[(lig_b, lig_a)]
                    sum_var += edge_uncertainty[(lig_b, lig_a)] ** 2
                else:
                    # Edge missing from network; skip this cycle
                    break

            else:
                # Normalize by sqrt(cycle length) to allow comparison across
                # different cycle lengths
                cc = abs(sum_ddgs / math.sqrt(len(cycle)))
                cc_uncertainty_normalized = abs(sum_ddgs) / math.sqrt(sum_var)
                rows.append(
                    {
                        "cycle": tuple(cycle),
                        "cc (kcal/mol)": round(cc, 2),
                        "cc_unc_normalized (kcal/mol)": round(cc_uncertainty_normalized, 2),
                    }
                )

        df = (
            pd.DataFrame(rows, columns=["cycle", "cc (kcal/mol)", "cc_unc_normalized (kcal/mol)"])
            .sort_values("cc (kcal/mol)", ascending=False)
            .reset_index(drop=True)
        )

        return df

    def get_cc_based_edge_statistics(self, max_cycle_length: int = 5) -> pd.DataFrame:
        """
        For each edge, report how many cycles it appears in and
        the mean and max cycle closure error of those cycles.
        Edges with high mean closure error across many cycles are
        likely candidates for re-simulation.

        Parameters
        ----------
        max_cycle_length : int, optional
            Only consider cycles up to this length. Defaults to 5.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'ligandA', 'ligandB', 'n_cycles',
            'mean_cc (kcal/mol)', 'max_cc (kcal/mol)', sorted by
            mean cycle closure error descending.
        """
        from collections import defaultdict

        cc_df = self.get_cycle_closure(max_cycle_length=max_cycle_length)
        network = self.to_legacy_graph()
        edge_ddg = {(a, b): d["calc_DDG"] for a, b, d in network.edges(data=True)}

        edge_cycles: dict[tuple, list[float]] = defaultdict(list)
        for _, row in cc_df.iterrows():
            cycle = list(row["cycle"])
            cc = row["cc (kcal/mol)"]
            for i, lig in enumerate(cycle):
                lig_a = lig
                lig_b = cycle[i + 1] if i < len(cycle) - 1 else cycle[0]
                edge = (lig_a, lig_b) if (lig_a, lig_b) in edge_ddg else (lig_b, lig_a)
                edge_cycles[edge].append(cc)

        rows = []
        for (a, b), ccs in edge_cycles.items():
            rows.append(
                {
                    "ligandA": a,
                    "ligandB": b,
                    "n_cycles": len(ccs),
                    "mean_cc (kcal/mol)": round(sum(ccs) / len(ccs), 3),
                    "max_cc (kcal/mol)": round(max(ccs), 3),
                }
            )

        return pd.DataFrame(rows).sort_values("mean_cc (kcal/mol)", ascending=False).reset_index(drop=True)
