cinnabar (formerly Arsenic)
==============================
[//]: # (Badges)

[![CI](https://github.com/OpenFreeEnergy/cinnabar/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenFreeEnergy/cinnabar/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/OpenFreeEnergy/cinnabar/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenFreeEnergy/cinnabar)
[![Documentation Status](https://readthedocs.org/projects/openff-cinnabar/badge/?version=latest)](https://openff-cinnabar.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6210305.svg)](https://doi.org/10.5281/zenodo.6210305)

# Reporting relative free energy results
Issue: we _must_ report statistics consistently and we would _like_ to plot these results consistently too

Solution: package that accepts relative free energy results reliably, which is untied to any particular method/system or software package. For this, the input should be as unconverted as possible.


### USAGE

`python cinnabar example.csv `

### OPTIONS

`python cinnabar --help`

### Terminology
D is difference (i.e. relative) while d is variance (i.e. error bar)
dDG would be the variance of an absolute FE, DDG would be the relative free energy between two molecules.

### Plots to output
There are two ways of thinking of the results of free energy simulations, one is as a method developer, where one cares about the distance of a simulation from the true experimental value. The other is as a drug designer - how does all the information of this method actually help me to pick which molecule to make next.
Statistics should definitely be printed on plots.
#### DDG’s
These should represent the primary data (i.e. for the method developer), output from the relative free energy simulations. There is still discussion to be had about the best way to report these. There are issues to decide as to

* Should we report only edges run or all edges

* Should we symmetrise

	If we only report edges that we run, it makes it harder to compare between results generated with different _sets_ of edges for the same system - I.e. if I run all the easy edges, I will look better than another method that has run more results. Plotting all edges gets around this, but moves us further from the primary data, and is somewhat redundant with the DG plot.

	Correlation statistics are variable based on the sign chosen for an edge, so if we are to report these, symmetrizing is the only way to make these robust. One solution would be to both not symmetrise and not report correlation statistics (only RMSE and AUE for these plots).

	If we are using these primary data plots, then it should very clear which edges are being plotted, so that we know if we are comparing one network to another or not. Maybe a networkX graph should be attached.

#### DG’s
These should represent the overall result (i.e. for the drug designer), where there relative free energies should be combined consistently (i.e. using MLE) to convert the available DDG’s into DG’s. As there can only be _Nligand_ data points on these plots, any statistics can be used, but possibly rank-ordering measures are most useful.

### Statistics
* RMSE - this is good

* MUE - this has issues when comparing between targets, as it is dependent on the dynamic range (noted by C. Bayly), but less so when comparing between methods. C. Bayly suggested Relative Absolute Error. Additionally, GRAM from GSK would be a good measure to incorporate  ([GRAM: A True Null Model for Relative Binding Affinity Predictions | Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00939))

* R2/Kendall etc (correlation coefficients) - there are issues of using these statistics with some DDG plots, and have more useful meaning with DG results (see 1examples/WhyNotToUseR2ForDDG.ipynb`)

### Errors
How do we compare errors? Several sources:
* MBAR

* Repeats (same simulation again)

* Repeats (forward/backward variety)

* Cycle closures

* Other sources (?)

We would like to handle these consistently. The input to the software should have two errors (a) generated from PYMBAR, as these are the de facto standard and (b) another column to contain other errors that may be generated, which may be used to try compensate for the underestimation of the MBAR errors.

### Plot styles - It may be impossible to completely agree on a plot style (and maybe not necessary)
Colours? Colourblind friendly?

Different colors for distance from equality (like David Hahn/de Groot lab)?

Error bars style?

Guidelines at _n_ units from equality?


#### TODO (move this to project board)
Generate set of plots that people are happy with
Add gram analysis for MUE
Incorporate edge errors into the bootstrapping?
Handle repeats properly
Handle forwards and backwards edges properly
Have entry point for absolute free energies too
Plots that look at other success metrics? i.e histogram of errors? (One like in METK?)
Currently just plotting everything against experimental, would like to do forcefield X vs. forcefield Y


### Copyright

Copyright (c) 2021, Hannah Bruce Macdonald


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
