
[![PyPI version](https://badge.fury.io/py/tick.svg)](https://badge.fury.io/py/tick)
[![Build Status](https://travis-ci.org/X-DataInitiative/tick.svg?branch=master)](https://travis-ci.org/X-DataInitiative/tick)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# tick

_tick_ is a Python 3 module for statistical learning, with a particular emphasis on time-dependent modeling. It is distributed under the 3-Clause BSD license, see [LICENSE.txt](LICENSE.txt).

The project was started in 2016 by Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas and Søren Vinther Poulsen at the [Datascience initiative](https://portail.polytechnique.edu/datascience/en) of [École Polytechnique](https://www.polytechnique.edu/en), France. The list of contributors is available in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Quick description

_tick_ is a machine learning library for Python 3. The focus is on statistical learning for time dependent systems, such as point processes. Tick features  also tools for generalized linear models and a generic optimization toolbox.
The core of the library is an optimization module providing model computational classes, solvers and proximal operators for regularization. It comes also with inference and simulation tools intended for end-users who for example can easily:

- Perform linear, logistic or Poisson regression
- Simulate point Hawkes processes with standard or exotic kernels. 
- Infer Hawkes models with various assumptions on the kernels: exponential or sum of exponential kernels, linear combination of basis kernels, sparse interactions, etc.

A comprehensive list of examples can be found at

- [https://x-datainitiative.github.io/tick/auto_examples/index.html](https://x-datainitiative.github.io/tick/auto_examples/index.html) 

and the documentation is available at

- [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)

The paper associated to this library has been published at

- [https://arxiv.org/abs/1707.03003](https://arxiv.org/abs/1707.03003)

If you use _tick_ in a scientific publication, we would appreciate citations.

<img align="right" src="https://raw.githubusercontent.com/X-DataInitiative/tick/master/doc/images/intel_logo.png" width="200" alt="intel logo" />
The <i>tick</i> library is released with the support of Intel®. It uses the Intel® Math Kernel Library (MKL) optimized for Intel® Xeon Phi™ and Intel® Xeon™ processors. <i>tick</i> runs efficiently on everything from desktop computers to powerful high-performance servers.


## Use cases

_tick_ is used for many industrial applications including:

* [A joint work](https://portail.polytechnique.edu/datascience/fr/node/329) with the French national social security (CNAMTS) to analyses a huge health-care database, that describes the medical care provided to most of the French citizens. For this project, _tick_ is used to detect weak signals in pharmacovigilance, in order quantify the impact of drugs exposures to the occurrence of adverse events.
   
* High-frequency order book modeling in finance, in order to understand the interactions between different event types and/or between different assets, leveraging the full time resolution available in the original data.

* Analyze the propagation of information in social media. Thanks to a dataset collected during 2017's presidential French election campaign on Twitter, _tick_ is used to recover, for each topic, the network across which information spreads inside the political sphere. 
  

## Quick setup

### Requirements

_tick_ currently works on Linux/OSX (Windows is experimental) systems and requires Python 3.4 or newer. Please have the required Python dependencies in your Python environment:

### Install using _pip_

_tick_ is available via _pip_. In your local Python environment (or global, with sudo rights), do:

    pip install tick

Installation may take a few minutes to build and link C++ extensions. At this point _tick_ should be ready to use available (if necessary, you can add _tick_ to the `PYTHONPATH` as explained below).

### Verify install

Run the following command and there should be no error

    python3 -c "import tick;"


## Source Installation

Please see the [INSTALL document](INSTALL.md)

## Help and Support

### Documentation

Documentation is available on 

- [https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick/)

This documentation is built with `Sphinx` and can be compiled and used locally by running `make html` from within the `doc` directory. This obviously needs to have `Sphinx` installed. Several tutorials and code-samples are available in the documentation.
 
### Communication

To reach the developers of _tick_, please join our community channel on Gitter (https://gitter.im/xdata-tick).

If you've found a bug that needs attention, please raise an issue here on Github. Please try to be as precise in the bug description as possible, such that the developers and other contributors can address the issue efficiently.

### Citation

If you use _tick_ in a scientific publication, we would appreciate citations. You can use the following bibtex entry:

    @ARTICLE{2017arXiv170703003B,
      author = {{Bacry}, E. and {Bompaire}, M. and {Ga{\"i}ffas}, S. and {Poulsen}, S.},
      title = "{tick: a Python library for statistical learning, with 
        a particular emphasis on time-dependent modeling}",
      journal = {ArXiv e-prints},
      eprint = {1707.03003},
      year = 2017,
      month = jul
    }


## Developers

Please see the [CONTRIBUTING document](CONTRIBUTING.md)
