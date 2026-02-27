"""
Estimators
==========

Estimators compute absolute free energy values from a set of relative
measurements stored in an FEMap.

"""
import abc
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING, List

import networkx as nx
import numpy as np

from cinnabar import stats
from cinnabar.measurements import Measurement, ReferenceState

if TYPE_CHECKING:
    from cinnabar.femap import FEMap


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
        e.g. ``"MLEEstimator"``.  Set automatically by `Estimator.estimate`.
    source : str
        The composed source label stamped on the output measurements,
        e.g. ``"MLE"`` for a single-source map or ``"MLE(openff-sage)"`` when
        multiple input sources are present.  Set automatically by
        `Estimator.estimate`.
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
        covariance_matrix
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
          estimator can use them to centre predictions.
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
                raise ValueError(
                    f"Computational results for source '{input_source}' are not fully connected"
                )
            # Only compose the label when it is actually needed to disambiguate.
            # Single-source users can then call get_estimator_metadata("MLE")
            # without having to know or construct the input source name.
            composed_source = (
                f"{self.source}({input_source})" if multiple_sources else self.source
            )

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
        ...


class MLEEstimator(Estimator):
    """Maximum-likelihood estimator (MLE) for absolute free energies.

    Uses the MLE solver from :mod:`cinnabar.stats` to compute the most
    probable set of absolute free energies consistent with the relative
    measurements stored in the map.

    Parameters
    ----------
    source : str, optional
        Label attached to the returned measurements and used as the storage
        key on the FEMap  Defaults to MLE.

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
        # TODO: replace stats.mle call with a self-contained implementation
        g, u = _build_graph_from_measurements(measurements)

        f_i_calc, C_calc = stats.mle(g, factor="calc_DDG")
        variance = np.diagonal(C_calc) ** 0.5

        ref = ReferenceState(label=source)
        ligand_order = list(g.nodes)

        out_measurements: List[Measurement] = []
        for n, f_i, df_i in zip(ligand_order, f_i_calc, variance):
            out_measurements.append(
                Measurement(
                    labelA=ref,
                    labelB=n,
                    DG=f_i * u,
                    uncertainty=df_i * u,
                    computational=True,
                    source=source,
                )
            )

        # anchor the estimator reference state to the global reference state
        out_measurements.append(
            Measurement(
                labelA=ReferenceState(),
                labelB=ref,
                DG=0.1 * u,
                uncertainty=0.0 * u,
                computational=True,
                source=source,
            )
        )

        return out_measurements, MLEEstimatorResult(
            covariance_matrix=C_calc,
            ligand_order=ligand_order,
        )



def _build_graph_from_measurements(
    measurements: List[Measurement],
) -> tuple[nx.DiGraph, object]:
    """Build a legacy graph from the list of measurements for use in the MLE method, this is copied over from the
    to_legacy_graph method of FEMap.

    Parameters
    ----------
    measurements : list[Measurement]
        Mix of relative computational and absolute experimental measurements.

    Returns
    -------
    g : nx.DiGraph
        Input graph ready for stats.mle
    u : unit
        The unit shared by all measurements (validated to be consistent).

    Raises
    ------
    ValueError
        If measurements have mixed units or duplicate computational edges
        exist between the same pair of nodes.
    """
    if not measurements:
        raise ValueError("No measurements provided")

    units = {m.DG.u for m in measurements}
    if len(units) > 1:
        raise ValueError(
            f"All measurements must share the same units before running an estimator. "
            f"Found: {units}"
        )
    u = next(iter(units))

    g = nx.DiGraph()
    edges_seen: List[tuple] = []

    for m in measurements:
        if not m.computational:
            continue
        if isinstance(m.labelA, ReferenceState):
            continue
        edge_name = tuple(sorted([m.labelA, m.labelB]))
        if edge_name in edges_seen:
            raise ValueError(
                f"Multiple edges detected between nodes {m.labelA} and {m.labelB}. "
                "MLE cannot be performed on graphs with multiple edges between the "
                "same nodes. The results should be combined into a single estimate "
                "and uncertainty before performing MLE. "
                "See https://cinnabar.openfree.energy/en/latest/concepts/estimators.html"
                "#limitations for more details."
            )
        g.add_edge(
            m.labelA, m.labelB,
            calc_DDG=m.DG.magnitude,
            calc_dDDG=m.uncertainty.magnitude,
        )
        edges_seen.append(edge_name)

    # annotate nodes with experimental absolute values
    for m in measurements:
        if m.computational:
            continue
        if not isinstance(m.labelA, ReferenceState):
            continue
        node = m.labelB
        if node not in g.nodes:
            continue
        g.nodes[node]["exp_DG"] = m.DG.magnitude
        g.nodes[node]["exp_dDG"] = m.uncertainty.magnitude
        g.nodes[node]["name"] = node

    # infer experimental DDG for edges where both endpoints have absolute data
    for A, B, d in g.edges(data=True):
        try:
            DG_A = g.nodes[A]["exp_DG"]
            dDG_A = g.nodes[A]["exp_dDG"]
            DG_B = g.nodes[B]["exp_DG"]
            dDG_B = g.nodes[B]["exp_dDG"]
        except KeyError:
            continue
        d["exp_DDG"] = DG_B - DG_A
        d["exp_dDDG"] = (dDG_A**2 + dDG_B**2) ** 0.5

    return g, u
