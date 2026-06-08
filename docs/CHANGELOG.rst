.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub cinnabar releases page <https://github.com/OpenFreeEnergy/cinnabar/releases>`_.

.. current developments

v0.6.0
====================

**Added:**

* Added the predictive index (PI) metric to compare ranking performance, this is also exposed in bootstrap statistics and can be used in plots `PR#194 <https://github.com/OpenFreeEnergy/cinnabar/pull/194>`_.
* Added the ability to define new estimators by subclassing the ``Estimator`` class and implementing the ``_estimate`` method, these can then be applied to the ``FEMap`` to generate absolute DG estimates. `PR#193 <https://github.com/OpenFreeEnergy/cinnabar/pull/193>`_.
* Added ``FEMap.get_cycle_closure_dataframe`` to calculate cycle closure errors for all cycles in the network,
  reporting raw closure errors, per-edge contributions, and uncertainty-normalized cycle closures. `PR#107 <https://github.com/OpenFreeEnergy/cinnabar/pull/107>`_.
* Added ``FEMap.get_cycle_closure_edge_statistics_dataframe`` to report per-edge cycle closure statistics. `PR#107 <https://github.com/OpenFreeEnergy/cinnabar/pull/107>`_.
* Added ``plotting.plot_cycle_closure`` to visualize the cycle closure error distribution as a histogram. `PR#107 <https://github.com/OpenFreeEnergy/cinnabar/pull/107>`_.
* Documentation and tutorials for the ECDF plotting functionality `PR#206 <https://github.com/OpenFreeEnergy/cinnabar/pull/206>`_.
* Added error estimates to ECDF plots via bootstrapping `PR#201 <https://github.com/OpenFreeEnergy/cinnabar/pull/201>`_.
* added ECDF plotting functionality to visualize the empirical cumulative distribution function of predicted vs experimental absolute, relative and all-to-all pairwise binding free energies `PR#172 <https://github.com/OpenFreeEnergy/cinnabar/pull/172>`_.
* Exposed stats calculation functions in the ``cinnabar.stats`` module to the public API and added docs allowing users to use them directly `PR#186 <https://github.com/OpenFreeEnergy/cinnabar/pull/186>`_.
* Added the ``compute_fraction_best_ligands`` function to compute the fraction of best ligands metric `PR#164 <https://github.com/OpenFreeEnergy/cinnabar/pull/164>`_.
* Added ``highlight_edges`` argument to the ``draw_graph`` function, allowing the user to highlight edges in the network graph `PR#203 <https://github.com/OpenFreeEnergy/cinnabar/pull/203>`_.
* Guidelines on scatter plots can now be set manually, the values are also annotated on the plots `PR#204 <https://github.com/OpenFreeEnergy/cinnabar/pull/204>`_.
* Added ``get_all_to_all_relative_dataframe()`` function the the ``FEMap`` class to compute pairwise relative free energy differences between all ligands in a dataset `PR#187 <https://github.com/OpenFreeEnergy/cinnabar/pull/187>`_.
* The ``get_relative/absolute/all_to_all_relative_dataframe()`` functions can now return values as ``pIC50``. This is controlled by passing ``observable_type="pic50"`` `PR#208 <https://github.com/OpenFreeEnergy/cinnabar/pull/208>`_.
* All scatter and ECDF plotting functions can now plot as ``pIC50``. This is controlled by passing ``observable_type="pic50"`` `PR#213 <https://github.com/OpenFreeEnergy/cinnabar/pull/213>`_.
* Added an affinity conversion function ``convert_observable`` which provides value and uncertainty conversions to and from ``dg``, ``ki``, ``ic50`` and ``pic50`` with units `PR#182 <https://github.com/OpenFreeEnergy/cinnabar/pull/182>`_.

**Changed:**

* The ``FEMMap.get_absolute_dataframe`` and ``FEMap.get_relative_dataframe`` methods now return dataframes in a sorted order of node labels grouped by source flags to enable easier comparisons between sources and with experimental values `PR#192 <https://github.com/OpenFreeEnergy/cinnabar/pull/192>`_.
* The ``FEMap.get_relative_dataframe`` now also includes experimental differences for the simulated legs in the dataframe to enable easier comparisons between predicted and experimental values `PR#192 <https://github.com/OpenFreeEnergy/cinnabar/pull/192>`_.
* ``cinnabar.arsenic`` has been renamed to ``cinnabar.cli``, other references to arsenic have also been removed. `PR#158 <https://github.com/OpenFreeEnergy/cinnabar/pull/158>`_.
* Plotting functions ``plot_DDGs``, ``plot_DGs`` and ``plot_all_DDGs`` now require an FEMap as input along with a computational source, see the API tutorial for more details `PR#212 <https://github.com/OpenFreeEnergy/cinnabar/pull/212>`_.
* The MLE estimator will now raise an error on any uncertainties of exactly zero on calculated edges due to issues with SVD calculations `PR#177 <https://github.com/OpenFreeEnergy/cinnabar/pull/177>`_.

**Deprecated:**

* ``plotting._master_plot`` is deprecated and will be removed in a future version. Use ``plotting.pair_plot`` instead `PR#204 <https://github.com/OpenFreeEnergy/cinnabar/pull/204>`_.
* ``FEMap.to_legacy_graph`` method is now deprecated and will be removed in a future release `PR#212 <https://github.com/OpenFreeEnergy/cinnabar/pull/212>`_.

**Fixed:**

* Fixed bug in ``FEMap.get_relative_dataframe()`` which gave the wrong values if absolute values were generated first using ``FEMap.generate_absolute_values()`` `PR#206 <https://github.com/OpenFreeEnergy/cinnabar/pull/206>`_.
* The ``plot_all_DDGs`` function now correctly uses the covariance matrix when available to compute error bars for the predicted values `PR#212 <https://github.com/OpenFreeEnergy/cinnabar/pull/212>`_.


v0.5.0
====================

**Added:**

* Added support for python 3.13.
* Vendored ``openff-models`` (PR `#131 <https://github.com/OpenFreeEnergy/cinnabar/pull/131>`_).
* Adding operator now available in ``FEMap``, i.e. you can add ``FEMap`` instances using ``+`` (PR `#114 <https://github.com/OpenFreeEnergy/cinnabar/pull/114>`_).


**Changed:**

* ``FEMap`` instances no longer allow ``.graph`` to be accessed directly, and instead use ``to_networkx()`` and ``from_networkx()`` methods (PR `#112 <https://github.com/OpenFreeEnergy/cinnabar/pull/112>`_).
* Scatter plots markers now have edge outlines for clearer visibility (PR `#113 <https://github.com/OpenFreeEnergy/cinnabar/pull/113>`_).


v0.2.1 - Release
====================

Bugfixes
^^^^^^^^
Fix erroneous MLE estimate when self-edges are included (PR `#38 <https://github.com/OpenFreeEnergy/cinnabar/pull/38>`_).
