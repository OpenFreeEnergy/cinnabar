============================
The ``FEMap`` Data Structure
============================

The :class:`.FEMap` is the **core abstraction in cinnabar**.
It provides a unified representation of free energy data, connecting the inputs from relative binding free energy
calculations with the **analyses** and **visualisations** that cinnabar enables.

Why a Graph?
------------

Relative free energy calculations are inherently **relational**: they compare two ligands at a time, producing a free
energy difference (:math:`\Delta\Delta G`) with an associated uncertainty. If we want to reason about a whole series of ligands, which
may be connected by multiple pairwise comparisons, we need to connect these edges in a logical way.

A **graph** is a natural way to represent this data, where:

- **Nodes** represent individual ligands.
- **Edges** represent pairwise free energy differences (:math:`\Delta\Delta G`) between ligands.
- **Absolute values** (:math:`\Delta G` experimental or calculated) can bet atached to nodes as attributes.

This graph representation is powerful: it allows integration of relative and absolute data, and it provides the
foundation for robust statistical analysis and visualization.


From Relative to Absolute
~~~~~~~~~~~~~~~~~~~~~~~~~

Although we calculate relative free energies directly, many applications require absolute binding free energies for
each ligand in order to rank the ligands for prioritization.

The :class:`.FEMap` provides convenience methods to support this transformation via a maximum likelihood estimation (MLE) [1]_
method by default. These methods take advantage of the entire network of relative free energies to infer absolute values.
Thus they require a graph which is at least weakly connected, that is there is a path between any two ligands in the graph.
A graph structure lends itself naturally to this type of analysis and the ``FEMap`` provides utilities to check and
visualise the connectivity of the network.


Redundancy and Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the strengths of a graph representation is that it can handle redundant data.
If multiple calculations connect the same ligands (e.g. forward and reverse transformations, repeats), they can easily be
incorperated into a directed graph structure, however not all estimators maybe able to take advantage of this information.

Cycles in the network provide consistency checks: if the sum of :math:`\Delta\Delta G` around a cycle is far from zero, that highlights systematic errors or poorly converged calculations.

Thus, the ``FEMap`` is not just a container, but also a tool for identifying inconsistencies in the underlying data.


References
~~~~~~~~~~~

.. [1] Xu, H., 2019. Optimal measurement network of pairwise differences. Journal of Chemical Information and Modeling, 59(11), pp.4720-4728.


