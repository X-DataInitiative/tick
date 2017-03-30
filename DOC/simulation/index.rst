
Simulation toolbox [`tick.simulation`]
======================================

.. _simu-intro:

Introduction
------------

`tick` provides several classes to simulate datasets. This is
particularly useful to test optimization algorithms, and to
compare the statistical properties of inference methods.

For now, `tick` gives simulation classes for some Generalized Linear
Models, Cox regression, Poisson Processes with any intensity and
Hawkes processes. Utilities for simulation of model coefficients
(with sparsity, etc.) and utilities for features matrix simulation
are provided as well.

Contents
--------

.. toctree::
   :maxdepth: 3

* :ref:`simu-intro`
* :ref:`simu-classes`
* :ref:`simu-utils`
* :ref:`simu-example`

.. _simu-classes:

Linear model simulation
-----------------------

====================  ======================
Model                 Class
====================  ======================
:ref:`simu-linreg`    `SimuLinReg`
:ref:`simu-logreg`    `SimuLogReg`
:ref:`simu-poisreg`   `SimuPoisReg`
:ref:`simu-coxreg`    `SimuCoxReg`
====================  ======================

Point process simulation
------------------------

=================================  ======================
Model                              Class
=================================  ======================
:ref:`simu-poisson`                `SimuPoissonProcess`
:ref:`simu-inhomogeneous-poisson`  `SimuInhomogeneousPoisson`
:ref:`simu-hawkes`                 `SimuHawkes`
=================================  ======================

.. _simu-utils:

Utilities
---------

Some utilities are provided to make simulation even simpler: simulation
of coefficients and simulation of features matrix. Note that the
in the model simulation classes (see below), features matrices can be
directly simulated or given by the user.

.. toctree::
   :maxdepth: 3

   utilities/features_matrix
   utilities/coefficients

.. _simu-example:

Example
-------
The next example shows how to simulate data from a Poisson regression
model, with the following characteristics:

* exponential link (default)
* simulated features matrix with Toeplitz covariance (default), with
  correlation=.8 between contiguous features
* sparse coefficients (10 non-zero coefficients)

Here is the sample code to do this::

    from tick.simulation import SimuPoisReg, weights_sparse_gauss

    weights0 = weights_sparse_gauss(200, 10, std=.1)
    intercept0 = 1.
    simu = SimuPoisReg(weights0, intercept0, n_samples=5000, cov_corr=0.8)
    X, y = simu.simulate()
    print(simu)
