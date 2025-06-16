.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub cinnabar releases page <https://github.com/OpenFreeEnergy/cinnabar/releases>`_.

0.5.0
=====

**Added:**

* Added support for python 3.13.
* Vendored ``openff-models`` (PR `#131 <https://github.com/OpenFreeEnergy/cinnabar/pull/131>`_).
* Adding operator now available in ``FEMap``, i.e. you can add ``FEMap`` instances using ``+`` (PR `#114 <https://github.com/OpenFreeEnergy/cinnabar/pull/114>`_).


**Changed:**

* ``FEMap`` instances no longer allow ``.graph`` to be accessed directly, and instead use ``to_networkx()`` and ``from_networkx()`` methods (PR `#112 <https://github.com/OpenFreeEnergy/cinnabar/pull/112>`_).
* Scatter plots markers now have edge outlines for clearer visibility (PR `#113 <https://github.com/OpenFreeEnergy/cinnabar/pull/113>`_).


0.2.1 - Release
---------------

Bugfixes
^^^^^^^^
Fix erroneous MLE estimate when self-edges are included (PR `#38 <https://github.com/OpenFreeEnergy/cinnabar/pull/38>`_).
