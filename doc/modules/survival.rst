

.. _survival:

====================
:mod:`tick.survival`
====================

This module provides some tools for survival analysis. It provides a learner for
Cox regression and a model for Self Control Case Series (SCCS).

1. Learners
============

For now, this module provides only a learner for Cox regression for proportional
hazards, using the partial likelihood. It also provides Nelson-Aalen and
Kaplan-Meier estimators.

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: function.rst

   CoxRegression
   nelson_aalen
   kaplan_meier
   ConvSCCS

2. Models
=========

The following models can be used with solvers for inference. Note that
:class:`ModelCoxRegPartialLik <tick.survival.ModelCoxRegPartialLik>` cannot
be used for now with a stochastic solver, while we can with :class:`ModelSCCS <tick.survival.ModelSCCS>`.

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelCoxRegPartialLik
   ModelSCCS

3. Simulation
=============

We provide tools for the simulation of datasets, for both right-censored
durations following a proportional risks model, and a SCCS model.
`SimuCoxReg` generates data in the form of i.i.d triplets :math:`(x_i, t_i, c_i)`
for :math:`i=1, \ldots, n`, where :math:`x_i \in \mathbb R^d` is a features vector,
:math:`t_i \in \mathbb R_+` is the survival time and :math:`c_i \in \{ 0, 1 \}` is the
indicator of right censoring.
Note that :math:`c_i = 1` means that :math:`t_i` is a failure time
while :math:`c_i = 0` means that :math:`t_i` is a censoring time.


.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuCoxReg
   SimuSCCS

Example
-------

.. plot:: ../examples/plot_simulation_coxreg.py
    :include-source:
