<h1 align="center">cinnabar (formerly Arsenic)</h1>

<p align="center">Tools to report and analyse relative free energy results</p>

<p align="center">
  <a href="https://github.com/OpenFreeEnergy/cinnabar/actions/workflows/ci.yml">
    <img alt="ci" src="https://github.com/OpenFreeEnergy/cinnabar/actions/workflows/ci.yml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/OpenFreeEnergy/cinnabar/main">
    <img alt="coverage" src="https://codecov.io/gh/OpenFreeEnergy/cinnabar/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://cinnabar.readthedocs.io/en/latest/?badge=latest">
    <img alt="license" src="https://app.readthedocs.org/projects/cinnabar/badge/?version=latest&style=flat" />
  </a>
  <a href="https://results.pre-commit.ci/latest/github/OpenFreeEnergy/cinnabar/main">
    <img alt="license" src="https://results.pre-commit.ci/badge/github/OpenFreeEnergy/cinnabar/main.svg" />
  </a>
  <a href="https://doi.org/10.5281/zenodo.6210305">
    <img alt="license" src="https://zenodo.org/badge/DOI/10.5281/zenodo.6210305.svg" />
  </a>
</p>

---


Issue: we _must_ report statistics consistently and we would _like_ to plot these results consistently too

Solution: The ``cinnabar`` package ensures a fair comparison of computational methods via robust and reproducible metrics implemented in the analysis functions following [standardized best practices](https://livecomsjournal.org/index.php/livecoms/article/view/v4i1e1497).
``cinnabar`` is designed with generality in mind, and can be used to analyse results from any free energy calculation software package.


## Installation

This package can be installed using `mamba`:

```shell
mamba install -c conda-forge cinnabar
```

## Getting started

To get started with ``cinnabar``, please see the [documentation](https://cinnabar.readthedocs.io/en/latest/).


### Copyright

Copyright (c) 2021, Hannah Bruce Macdonald


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
