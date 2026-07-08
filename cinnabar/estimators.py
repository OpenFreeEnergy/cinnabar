"""
Estimators
==========

Estimators compute absolute free energy values from a set of relative
measurements stored in an FEMap.

"""

import abc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from openff.units import Quantity

from cinnabar import stats
from cinnabar._due import Doi, due
from cinnabar.measurements import Measurement, ReferenceState

if TYPE_CHECKING:
    from cinnabar.femap import FEMap  # pragma: no cover

due.cite(
    Doi("10.1021/acs.jcim.9b00528"),
    description="Compute maximum likelihood estimate of free energies and covariance in their estimates",
    path="cinnabar.estimators.MLEEstimator.mle",
    cite_module=True,
)


@dataclass
class EstimatorResult:
    """Simple base class for estimator results with provenance fields.

    This is the base class for all estimator results and should be used to
    store any additional metadata that is not appropriate for the returned
    measurements but might be useful for downstream analysis, like the
    covariance matrix of the MLE estimator.  Subclasses should define
    specific typed fields for this metadata to help IDEs.

    Attributes
    ----------
    estimator : str
        The class name of the estimator that produced this result,
        e.g. ``"MLEEstimator"``.  Set automatically by ``Estimator.estimate``.
    source : str
        The composed source label stamped on the output measurements,
        e.g. ``"MLE"`` for a single-source map or ``"MLE(openff-sage)"`` when
        multiple input sources are present.  Set automatically by
        ``Estimator.estimate``.
    """

    estimator: str = field(default="", init=False)
    source: str = field(default="", init=False)


@dataclass
class MLEEstimatorResult(EstimatorResult):
    """Results data produced by `MLEEstimator`.

    Attributes
    ----------
    covariance_matrix : np.ndarray, shape (N, N)
        Full MLE covariance matrix.  Entry ``[i, j]`` is the covariance
        between the free energy estimates of ligands ``i`` and ``j``.

    ligand_order : list
        Ordered list of ligand labels whose index maps to rows/columns of
        ``covariance_matrix``
    """

    covariance_matrix: np.ndarray
    ligand_order: list


class Estimator(abc.ABC):
    """Abstract base class for free-energy estimators.

    Subclasses must implement the ``_estimate`` method and set a ``source`` class
    attribute that is used as the ``source`` field on returned
    ``Measurement`` objects and as the key under
    which the ``EstimatorResult`` is stored on the FEMap.
    """

    source: str

    @staticmethod
    def _check_weakly_connected(measurements: list[Measurement]) -> bool:
        """Check if the computational graph of the provided measurements is connected."""
        g = nx.MultiGraph()
        for m in measurements:
            if m.computational and not isinstance(m.labelA, ReferenceState):
                g.add_edge(m.labelA, m.labelB)
        try:
            return nx.is_connected(g)
        except nx.NetworkXPointlessConcept:
            return False

    def estimate(self, femap: "FEMap") -> dict[str, tuple[list[Measurement], EstimatorResult]]:
        """Run the estimator on the FEMap for each unique computational source.

        Parameters
        ----------
        femap : FEMap
            The map to estimate from.

        Returns
        -------
        dict[str, tuple[list[Measurement], EstimatorResult]]
            A dictionary mapping the *composed source label* to a
            ``(measurements, result)`` tuple.  The composed label is
            ``"{estimator.source}({input_source})"`` when the FEMap contains
            more than one computational source (e.g. ``"MLE(openfe)"``), or
            just ``"{estimator.source}"`` when there is only one, so that
            single-source users never need to know the input source name.

        Notes
        -----
        * Connectivity is checked per source before ``_estimate`` is called.
        * Experimental measurements are forwarded to every source so the
          estimator can use them to center predictions.
        * The estimates are stamped with a composed source label of the form ``"{estimator.source}({input_source})"``
            when multiple computational sources are present, or just ``"{estimator.source}"`` when there is only one.
        """
        measurements_by_source: dict[str, list[Measurement]] = defaultdict(list)
        experimental_measurements: list[Measurement] = []

        for m in femap:
            if m.computational:
                measurements_by_source[m.source].append(m)
            else:
                experimental_measurements.append(m)

        multiple_sources = len(measurements_by_source) > 1

        results = {}
        for input_source, comp_measurements in measurements_by_source.items():
            if not self._check_weakly_connected(comp_measurements):
                raise ValueError(f"Computational results for source '{input_source}' are not fully connected")
            # Only compose the label when it is actually needed to disambiguate.
            # Single-source users can then call get_estimator_metadata("MLE")
            # without having to know or construct the input source name.
            composed_source = f"{self.source}({input_source})" if multiple_sources else self.source

            measurements, result = self._estimate(
                comp_measurements + experimental_measurements,
                source=composed_source,
            )

            # Stamp provenance automatically so subclasses don't have to.
            result.estimator = type(self).__name__
            result.source = composed_source
            results[composed_source] = (measurements, result)

        return results

    @abc.abstractmethod
    def _estimate(
        self,
        measurements: list[Measurement],
        source: str,
    ) -> tuple[list[Measurement], EstimatorResult]:
        """Estimate absolute free energies from a list of measurements.

        Measurements can be a mix of computational and experimental relative and absolute free energy measurements.
        Absolute values should be used to center the results if possible.

        Parameters
        ----------
        measurements : list[Measurement]
            A list of absolute and relative free energy measurements to estimate from this can include both computational and
            experimental values.
        source : str
            The composed source label to stamp on returned measurements and use as the key for storing the result on the FEMap.

        Returns
        -------
        measurements : list[Measurement]
            Absolute free energy estimates to be added to the FEMap.
        result : EstimatorResult
            Estimator-specific intermediate data that cannot be reconstructed
            from the measurements alone.

        Raises
        ------
        ValueError
            If the estimator cannot be applied (e.g. the graph is not
            connected, or there are duplicate edges).
        """
        ...  # pragma: no cover


class MLEEstimator(Estimator):
    """Maximum-likelihood estimator (MLE) for absolute free energies.

    Uses the MLE solver from :mod:`cinnabar.stats` to compute the most
    probable set of absolute free energies consistent with the relative
    measurements stored in the map.

    Parameters
    ----------
    source : str, default "MLE"
        Label attached to the returned measurements and used as the storage
        key on the FEMap. Defaults to MLE.

    Notes
    -----
    * Requires the computational sub-graph to be weakly connected.
    * Cannot handle multiple edges between the same pair of nodes; combine
      replicates into a single estimate before calling this estimator.
    """

    def __init__(self, source: str = "MLE"):
        self.source = source

    def _estimate(
        self,
        measurements: list[Measurement],
        source: str,
    ) -> tuple[list[Measurement], MLEEstimatorResult]:
        """Run MLE on the measurements and return the estimated DG values.

        Parameters
        ----------
        measurements : list[Measurement]
            Relative computational edges plus any experimental or computational absolute
            measurements for a single source.
        source : str
            The composed source label to stamp on returned measurements and use as the key for storing the result on the FEMap.

        Returns
        -------
        measurements : list[Measurement]
            One absolute-DG ``Measurement`` per ligand, plus an anchor
            connecting the MLE reference state to the global
            ``ReferenceState``.
        result : MLEEstimatorResult
            Contains :attr:`~MLEEstimatorResult.covariance_matrix` and
            :attr:`~MLEEstimatorResult.ligand_order`.
        """
        graph, unit = self._build_graph_from_measurements(measurements)

        f_i_calc, C_calc = self.mle(graph, factor="calc_DDG")
        variance = np.diagonal(C_calc) ** 0.5

        ref = ReferenceState(label=source)
        ligand_order = list(graph.nodes)

        out_measurements: list[Measurement] = []
        for label, f_i, df_i in zip(ligand_order, f_i_calc, variance):
            out_measurements.append(
                Measurement(
                    labelA=ref,
                    labelB=label,
                    DG=f_i * unit,
                    uncertainty=df_i * unit,
                    computational=True,
                    source=source,
                )
            )

        # anchor the estimator reference state to the global reference state
        out_measurements.append(
            Measurement(
                labelA=ReferenceState(),
                labelB=ref,
                DG=Quantity(0.1, units=unit),
                uncertainty=Quantity(0.0, units=unit),
                computational=True,
                source=source,
            )
        )

        return out_measurements, MLEEstimatorResult(
            covariance_matrix=C_calc,
            ligand_order=ligand_order,
        )

    @staticmethod
    def mle(graph: nx.DiGraph, factor: str = "f_ij", node_factor: str | None = None) -> (np.ndarray, np.ndarray):
        """
        Compute maximum likelihood estimate of free energies and covariance in their estimates.
        The number 'factor' is the node attribute on which the MLE will be calculated,
        where d'factor' will be used as the standard error of the factor

        Reference : https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528
        Xu, Huafeng. "Optimal measurement network of pairwise differences."
        Journal of Chemical Information and Modeling 59.11 (2019): 4720-4728.

        NOTE: Self-edges (edges that connect a node to itself) will be ignored.

        Parameters
        ----------
        graph :nx.Graph
            The graph for which an estimate is to be computed
            Each edge must have attributes 'f_ij' and 'df_ij' for the free energy and uncertainty
            estimate
            Will have 'bayesian_f_ij' and 'bayesian_df_ij' added to each edge
            and 'bayesian_f_i' and 'bayesian_df_i' added to each node.
        factor : string, default = 'f_ij'
            node attribute of nx.Graph that will be used for MLE
        node_factor : string, default = None
            optional - provide if there is node data (i.e. absolute values) 'f_i' or 'exp_DG' to
            include will expect a corresponding uncertainty 'f_di' or 'exp_dDG'
        Returns
        -------
        f_i : np.array with shape (n_ligands,)
            f_i[i] is the absolute free energy of ligand i in kcal/mol

        C : np.array with shape (n_ligands, n_ligands)
            C[i,j] is the covariance of the free energy estimates of i and j

        """
        # if we have bidirectional edge results we need to raise an error as they can not be used with MLE
        # track the edges we have seen
        edges = []
        for a, b in graph.edges:
            if (edge_name := (a, b) if str(a) < str(b) else (b, a)) in edges:
                # TODO this should be supported behavior
                raise ValueError(
                    f"Multiple edges detected between nodes {a} and {b}. MLE cannot be performed on graphs with multiple "
                    f"edges between the same nodes. The results should be combined into a single estimate and uncertainty "
                    f"before performing MLE. See https://cinnabar.openfree.energy/en/latest/concepts/estimators.html#limitations for more details."
                )
            edges.append(edge_name)

        n_nodes = graph.number_of_nodes()

        node_label = None if node_factor is None else node_factor.replace("_", "_d")
        node_name_to_index = {name: i for i, name in enumerate(graph.nodes())}
        # Adapted from the multibind implementation (Kenney IM & Beckstein O, 2023)
        # https://github.com/Becksteinlab/multibind/blob/7c93f605d99ff67d9adef890c9302bccd2caa1b5/multibind/multibind.py#L319
        # to support harmonic wells around individual states
        z = np.zeros((n_nodes,))
        F_matrix = np.zeros((n_nodes, n_nodes))

        for n, data in graph.nodes(data=True):
            if node_label in data:
                i = node_name_to_index[n]
                z[i] = data[node_factor] / (data[node_label] ** 2)
                F_matrix[i,i] = 1 / (data[node_label] ** 2)

        for a, b, data in graph.edges(data=True):

            if a == b:
                continue

            i = node_name_to_index[a]
            j = node_name_to_index[b]

            deltaij = data[factor]
            if (varij := data[factor.replace("_", "_d")] ** 2) == 0:
                raise ValueError(f"MLE solver will fail with zero reported uncertainty for calculated differences. Edge ({a}, {b}) has zero uncertainty check inputs.")

            z[i] += -deltaij / varij
            z[j] += deltaij / varij

            F_matrix[i, i] += 1 / varij
            F_matrix[j, j] += 1 / varij
            F_matrix[i, j] += -1 / varij
            F_matrix[j, i] += -1 / varij

        Finv = np.linalg.pinv(F_matrix, hermitian=True)
        f_i = np.matmul(Finv, z)
        return f_i, Finv

    @staticmethod
    def form_edge_matrix(graph: nx.Graph, label: str, step=None, action=None, node_label=None) -> np.ndarray:
        """
        Extract the labeled property from edges into a matrix.

        Parameters
        ----------
        graph : nx.Graph
            The graph to extract data from
        label : str
            The label to use for extracting edge properties
        action : str, optional, default=None
            If 'symmetrize', returns a symmetric matrix A[i,j] = A[j,i]
            If 'antisymmetrize', returns an antisymmetric matrix A[i,j] = -A[j,i]
        node_label : sr, optional, default=None
            Diagonal will be occupied with absolute values, where labelled

        Returns
        ----------
        matrix
        """
        N = len(graph.nodes)
        matrix = np.zeros([N, N])

        node_name_to_index = {}
        for i, name in enumerate(graph.nodes()):
            node_name_to_index[name] = i

        for a, b in graph.edges:
            i = node_name_to_index[a]
            j = node_name_to_index[b]
            matrix[j, i] = graph.edges[a, b][label]
            if action == "symmetrize":
                matrix[i, j] = matrix[j, i]
            elif action == "antisymmetrize":
                matrix[i, j] = -matrix[j, i]
            elif action is None:
                pass
            else:
                raise ValueError(f'action "{action}" unknown.')

        if node_label is not None:
            for n in graph.nodes(data=True):
                i = node_name_to_index[n[0]]
                if node_label in n[1]:
                    matrix[i, i] = n[1][node_label]

        return matrix

    @staticmethod
    def _build_graph_from_measurements(
        measurements: list[Measurement],
    ) -> tuple[nx.DiGraph, object]:
        """Build a legacy graph from the list of measurements for use in the MLE method, this is copied over from the
        to_legacy_graph method of FEMap.

        Parameters
        ----------
        measurements : list[Measurement]
            Mix of relative computational and absolute experimental measurements.

        Returns
        -------
        graph : nx.DiGraph
            Input graph ready for stats.mle
        unit : unit
            The unit shared by all measurements (validated to be consistent).

        Raises
        ------
        ValueError
            If measurements have mixed units or duplicate computational edges
            exist between the same pair of nodes.
        """
        if not measurements:
            raise ValueError("No measurements provided")

        if len(units := {m.DG.u for m in measurements}) > 1:
            raise ValueError(f"All measurements must share the same units before running an estimator. Found: {units}")
        unit = units.pop()

        graph = nx.DiGraph()
        edges_seen: List[tuple] = []

        # populate the edges of the graph along with their computational binding free energies
        for m in filter(lambda m: m.computational, measurements):
            if isinstance(m.labelA, ReferenceState):
                # TODO this is never hit in the tests and should be supported behavior
                continue
            edge_name = (m.labelA, m.labelB) if str(m.labelA) < str(m.labelB) else (m.labelB, m.labelA)
            if edge_name in edges_seen:
                # TODO this is a limitation of the software, not the method. Support for multiple edges should be a priority
                raise ValueError(
                    f"Multiple edges detected between nodes {m.labelA} and {m.labelB}. "
                    "MLE cannot be performed on graphs with multiple edges between the "
                    "same nodes. The results should be combined into a single estimate "
                    "and uncertainty before performing MLE. "
                    "See https://cinnabar.openfree.energy/en/latest/concepts/estimators.html"
                    "#limitations for more details."
                )
            graph.add_edge(
                m.labelA,
                m.labelB,
                calc_DDG=m.DG.magnitude,
                calc_dDDG=m.uncertainty.magnitude,
            )
            edges_seen.append(edge_name)

        # annotate nodes with experimental absolute values, this doesn't add edges
        for m in filter(lambda m: not m.computational, measurements):
            # labelA must always be a reference state, otherwise it is ignored
            if not isinstance(m.labelA, ReferenceState):
                # TODO this is never hit in the tests
                continue
            # TODO to support experimental values, we need this to not be true
            # do not include experimental information if no computation data is already present
            if (node := m.labelB) not in graph.nodes:
                continue
            graph.nodes[node]["exp_DG"] = m.DG.magnitude
            graph.nodes[node]["exp_dDG"] = m.uncertainty.magnitude
            graph.nodes[node]["name"] = node

        return graph, unit
