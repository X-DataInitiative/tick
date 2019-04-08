
.. _simulation:

======================
:mod:`tick.simulation`
======================

This module provides basic tools for the simulation of model weights and
features matrices.
This is particularly useful to test optimization algorithms, and to
compare the statistical properties of inference methods.

1. Simulation of model weights
==============================

Here are functions for the simulation of model weights.

.. currentmodule:: tick.simulation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   weights_sparse_exp
   weights_sparse_gauss

**Example**

.. plot:: modules/code_samples/plot_simulation_weights.py
    :include-source:

2. Simulation of a features matrix
==================================

Here are functions for the simulation of a features matrix: each simulated
vector or features is distributed as a centered Gaussian vector with
a particular covariance matrix (uniform symmetrized or toeplitz).

.. currentmodule:: tick.simulation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   features_normal_cov_uniform
   features_normal_cov_toeplitz
