============================
The ``FEMap`` Data Structure
============================

The :class:`.FEMap` is the **core abstraction in cinnabar**.
It provides a unified representation of free energy data, connecting the inputs from relative binding free energy
calculations with the **analyses** and **visualizations** that cinnabar enables.

Why a Graph?
------------

Relative free energy calculations are inherently **relational**: they compare two ligands at a time, producing a free
energy difference (:math:`\Delta\Delta G`) with an associated uncertainty. If we want to reason about a whole series of ligands, which
may be connected by multiple pairwise comparisons, we need to connect these edges in a logical way.

A **graph** is a natural way to represent this data, where:

- **Nodes** represent individual ligands.
- **Absolute values** (:math:`\Delta G` experimental or calculated) can be atached to nodes as attributes.
- **Edges** represent pairwise free energy differences (:math:`\Delta\Delta G`) between ligands.

This graph representation is powerful: it allows integration of relative and absolute data, and it provides the
foundation for robust statistical analysis and visualization.


From Relative to Absolute
~~~~~~~~~~~~~~~~~~~~~~~~~

Although we calculate relative free energies directly, many applications require absolute binding free energies for
each ligand in order to rank the ligands for prioritization.

The :class:`.FEMap` provides convenience methods to support this transformation via a :ref:`maximum likelihood estimation (MLE) <estimators>` [1]_
method by default. These methods take advantage of the entire network of relative free energies to infer absolute values.
Thus they require a graph which is at least weakly connected, with at least one path between any two ligands in the graph.
The ``FEMap`` provides utilities to check (:meth:`.FEMap.check_weakly_connected`) and visualise (:meth:`.FEMap.draw_graph`) the connectivity of the network.


References
~~~~~~~~~~~

.. [1] Xu, H., 2019. Optimal measurement network of pairwise differences. Journal of Chemical Information and Modeling, 59(11), pp.4720-4728.


