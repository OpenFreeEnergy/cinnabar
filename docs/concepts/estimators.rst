.. _estimators:

===============================
Absolute Free Energy Estimators
===============================

Relative free energy calculations produce :math:`\Delta\Delta G` (differences between two ligands). To compare these with experiment,
or to rank ligands by affinity, we need absolute free energies (:math:`\Delta G`) for each ligand.

This requires an **estimator**: a method that takes the network of relative free energies and produces absolute values.


Maximum Likelihood Estimation (MLE)
-----------------------------------

The Maximum Likelihood Estimation (MLE) [1]_ method is the **default** estimator used in cinnabar to obtain absolute
free energies (:math:`\Delta G`) from a network of relative free energies (:math:`\Delta\Delta G`).


The Core Idea
~~~~~~~~~~~~~~
To place every ligand on a common absolute scale, we need to find a set of :math:`\Delta G` values that best explain all
relative differences (:math:`\Delta\Delta G`) simultaneously. The MLE method does this by asking:

 What set of :math:`\Delta G` values makes the observed data most likely, given the reported uncertainties?

This framing naturally integrates all edges and cycles in the graph simultaneously.

The Likelihood Function
~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have a network with two ligands ``i`` and ``j``, with observed relative free energy :math:`\Delta\Delta G_{ij}` and
uncertainty :math:`\sigma_{ij}`. The model assumes each measurement is normally distributed:

.. math::

   \Delta\Delta G_{ij} \approx \mathcal{N}(\Delta G_j - \Delta G_i, \sigma_{ij}^2)

The **likelihood** is the product of probabilities for all edges in the graph. The MLE procedure finds the set of :math:`\Delta G`
values that maximises this likelihood (or equivalently, minimises the negative log-likelihood).

Uncertainty Propagation
~~~~~~~~~~~~~~~~~~~~~~~

Input uncertainties (:math:`\sigma_{ij}`) are explicitly included in the likelihood function. This means more precise edges
(smaller uncertainty) have greater weight in determining the solution. However, this does mean that high confidence but low
accuracy edges can impact the entire network and so robust uncertainty estimates on input data are crucial.


Centering of Results
~~~~~~~~~~~~~~~~~~~~

The absolute :math:`\Delta G` scale is arbitrary: adding a constant to all :math:`\Delta G` values does not change any relative differences :math:`\Delta\Delta G`.
As a result, the MLE solution is typically centred around zero (or another chosen reference). To compare with experimental
values, an experimental shift must be applied. By default ``cinnabar`` will align the mean of predicted and
experimental :math:`\Delta G` in the plotting functions.


Limitations
~~~~~~~~~~~

- The MLE method **can not** use multiple independent measurements of the same edge to improve precision automatically.
  Each edge must be represented by a single :math:`\Delta\Delta G` and uncertainty. If multiple measurements are available,
  they should be combined externally (e.g. via weighted averaging) before input to the estimator.



References
~~~~~~~~~~~

.. [1] Xu, H., 2019. Optimal measurement network of pairwise differences. Journal of Chemical Information and Modeling, 59(11), pp.4720-4728.
