
[![PyPI version](https://badge.fury.io/py/tick.svg)](https://badge.fury.io/py/tick)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

| Operating system  | Build Status |
| ------------- | ------------- |
| Linux  | [![Linux Build](https://github.com/X-DataInitiative/tick/actions/workflows/build_nix.yml/badge.svg?branch=master)](https://github.com/X-DataInitiative/tick/actions/workflows/build_nix.yml)  |
| macOS  | [![macOS Build](https://github.com/X-DataInitiative/tick/actions/workflows/build_osx.yml/badge.svg?branch=master)](https://github.com/X-DataInitiative/tick/actions/workflows/build_osx.yml)  |
| Windows  | [![Windows Build](https://github.com/X-DataInitiative/tick/actions/workflows/build_win.yml/badge.svg?branch=master)](https://github.com/X-DataInitiative/tick/actions/workflows/build_win.yml)  |


# tick

_tick_ is a Python 3 module for statistical learning, with a particular emphasis on time-dependent modeling. It is distributed under the 3-Clause BSD license, see [LICENSE.txt](LICENSE.txt).

The project was started in 2016 by Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas and Søren Vinther Poulsen at the Data Science Initiative of [École Polytechnique](https://www.polytechnique.edu/en), France. The list of contributors is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Quick description

_tick_ is a machine learning library for Python 3. The focus is on statistical learning for time dependent systems, such as point processes. _tick_ also features tools for generalized linear models and a generic optimization toolbox.
The core of the library is an optimization module providing model computational classes, solvers and proximal operators for regularization. It also comes with inference and simulation tools intended for end-users who for example can easily:

- Perform linear, logistic or Poisson regression
- Simulate Hawkes processes with standard or exotic kernels.
- Infer Hawkes models with various assumptions on the kernels: exponential or sum of exponential kernels, linear combination of basis kernels, sparse interactions, etc.

- Documentation: [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)
- Examples gallery: [https://x-datainitiative.github.io/tick/auto_examples/index.html](https://x-datainitiative.github.io/tick/auto_examples/index.html)
- Paper: [Journal of Machine Learning Research 18(214), 2018](https://jmlr.org/papers/v18/17-381.html) (preprint: [arXiv:1707.03003](https://arxiv.org/abs/1707.03003))

If you use _tick_ in a scientific publication, we would appreciate citations (see [Citation](#citation) below).

<img align="right" src="https://raw.githubusercontent.com/X-DataInitiative/tick/master/doc/images/intel_logo.png" width="200" alt="intel logo" />
The <i>tick</i> library was released with the support of Intel®. <i>tick</i> runs efficiently on everything from desktop computers to powerful high-performance servers.


## Use cases

_tick_ is used for many industrial applications including:

* A joint work with the French national health insurance (CNAMTS) to analyze a huge health-care database that describes the medical care provided to most French citizens. For this project, _tick_ is used to detect weak signals in pharmacovigilance, in order to quantify the impact of drug exposures to the occurrence of adverse events.

* High-frequency order book modeling in finance, in order to understand the interactions between different event types and/or between different assets, leveraging the full time resolution available in the original data.

* Analyzing the propagation of information in social media. Thanks to a dataset collected during 2017's presidential French election campaign on Twitter, _tick_ is used to recover, for each topic, the network across which information spreads inside the political sphere.


## Quick setup

### Requirements

_tick_ requires Python 3.11 or newer, with `numpy>=2.0` and `scipy>=1.13`. Pre-built wheels are published on PyPI for Linux, macOS and Windows.

### Install using _pip_

In your Python environment (a virtualenv is recommended), run:

    pip install tick

### Verify install

The following command should run without error:

    python3 -c "import tick"


## Source Installation

Please see the [INSTALL document](INSTALL.md)

## Help and Support

### Documentation

Documentation is available on

- [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)

This documentation is built with `Sphinx` and can be compiled and used locally by running `make html` from within the `doc` directory. This obviously needs to have `Sphinx` installed. Several tutorials and code-samples are available in the documentation.

### Communication

For questions, bug reports and feature requests, please open an [issue on GitHub](https://github.com/X-DataInitiative/tick/issues). Please be as precise as possible in the description (ideally with a minimal reproducible example), so that the developers and other contributors can address it efficiently.

### Citation

If you use _tick_ in a scientific publication, we would appreciate citations. You can use the following bibtex entry:

    @article{JMLR:v18:17-381,
      author  = {Emmanuel Bacry and Martin Bompaire and Philip Deegan and St{\'e}phane Ga{\"i}ffas and S{\o}ren V. Poulsen},
      title   = {tick: a Python Library for Statistical Learning, with an emphasis on Hawkes Processes and Time-Dependent Models},
      journal = {Journal of Machine Learning Research},
      year    = {2018},
      volume  = {18},
      number  = {214},
      pages   = {1--5},
      url     = {http://jmlr.org/papers/v18/17-381.html}
    }


## Developers

Please see the [CONTRIBUTING document](CONTRIBUTING.md)
