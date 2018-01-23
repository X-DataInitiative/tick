:orphan:

.. _api:

=============
API Reference
=============

This is the full class and function references of tick. Please look at
the modules documentation cited below for more examples and use cases,
since direct class and function API is not enough for understanding their uses.

Table of contents
=================

* :ref:`tick.hawkes <api-hawkes>`
* :ref:`tick.linear_model <api-linear_model>`
* :ref:`tick.robust <api-robust>`
* :ref:`tick.survival <api-survival>`
* :ref:`tick.prox <api-prox>`
* :ref:`tick.solver <api-solver>`
* :ref:`tick.hawkes.simulation <api-simulation>`
* :ref:`tick.plot <api-plot>`
* :ref:`tick.datasets <api-datasets>`
* :ref:`tick.preprocessing <api-preprocessing>`
* :ref:`tick.metrics <api-metrics>`

.. _api-hawkes:

:ref:`hawkes`
=============

This module provides tools for the inference and simulation of Hawkes processes.

**User guide:** :ref:`hawkes`

Learners
--------

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesExpKern
   HawkesSumExpKern
   HawkesEM
   HawkesADM4
   HawkesBasisKernels
   HawkesSumGaussians
   HawkesConditionalLaw
   HawkesCumulantMatching

Simulation
----------

Time function
*************

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.TimeFunction

Hawkes kernels
**************

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesKernelExp
   HawkesKernelSumExp
   HawkesKernelPowerLaw
   HawkesKernelTimeFunc

Simulation of point processes
*****************************

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst


   SimuPoissonProcess
   SimuInhomogeneousPoisson
   SimuHawkes
   SimuHawkesExpKernels
   SimuHawkesSumExpKernels
   SimuHawkesMulti

Models
------

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHawkesExpKernLogLik
   ModelHawkesExpKernLeastSq
   ModelHawkesSumExpKernLogLik
   ModelHawkesSumExpKernLeastSq


.. _api-linear_model:

:ref:`linear_model`
===================

This modules provides tools for the inference and simulation of generalized
linear models.

**User guide:** :ref:`linear_model`

Learners
--------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LinearRegression
   LogisticRegression
   PoissonRegression

Models
------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelLinReg
   ModelLogReg
   ModelPoisReg
   ModelHinge
   ModelSmoothedHinge
   ModelQuadraticHinge

Simulation
----------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuLinReg
   SimuLogReg
   SimuPoisReg


.. _api-robust:

:ref:`robust`
=============

This module provides tools for robust inference, namely outliers detection
and models such as Huber regression, among others robust losses.

**User guide:** :ref:`robust`

Tools for robust inference and outliers detection
-------------------------------------------------

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: function.rst

   RobustLinearRegression
   std_mad
   std_iqr

Robust losses
-------------

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHuber
   ModelModifiedHuber
   ModelAbsoluteRegression
   ModelEpsilonInsensitive
   ModelLinRegWithIntercepts


.. _api-survival:

:ref:`survival`
===============

This module provides tools for inference and simulation for survival analysis.

**User guide:** :ref:`survival`

Inference
---------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: function.rst

   CoxRegression
   nelson_aalen
   kaplan_meier

Models
------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelCoxRegPartialLik
   ModelSCCS

Simulation
----------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuCoxReg


.. _api-prox:

:ref:`prox`
===========

This module contains all the proximal operators available in tick.

**User guide:** See the :ref:`prox` section for further details.

.. automodule:: tick.prox
   :no-members:
   :no-inherited-members:

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   prox.ProxZero
   prox.ProxL1
   prox.ProxL1w
   prox.ProxElasticNet
   prox.ProxL2Sq
   prox.ProxL2
   prox.ProxMulti
   prox.ProxNuclear
   prox.ProxPositive
   prox.ProxEquality
   prox.ProxSlope
   prox.ProxTV
   prox.ProxBinarsity
   prox.ProxGroupL1


.. _api-solver:

:ref:`solver`
=============

This module contains all the solvers available in tick.

**User guide:** See the :ref:`solver` section for further details.

.. automodule:: tick.solver
   :no-members:
   :no-inherited-members:

Batch solvers
-------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.GD
   solver.AGD
   solver.BFGS
   solver.GFB
   solver.SCPG

Stochastic solvers
------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.SGD
   solver.AdaGrad
   solver.SVRG
   solver.SAGA
   solver.SDCA

History
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.History


.. _api-simulation:

:ref:`simulation`
=================

This module contains basic tools from simulation.

**User guide:** See the :ref:`simulation` section for further details.

Weights simulation
------------------

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss

Features simulation
-------------------

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz


.. _api-plot:

:ref:`plot`
===========

This module contains some utilities functions for plotting.

**User guide:** See the :ref:`plot` section for further details.

Functions
---------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_history
   plot.plot_hawkes_kernels
   plot.plot_hawkes_kernel_norms
   plot.plot_basis_kernels
   plot.plot_timefunction
   plot.plot_point_process
   plot.stems


.. _api-datasets:

:ref:`dataset`
==============

This module provides easy access to some datasets used as benchmarks in `tick`.

**User guide:** See the :ref:`dataset` section for further details.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.fetch_tick_dataset
   dataset.fetch_hawkes_bund_data


.. _api-preprocessing:

:ref:`preprocessing`
====================

This module contains some utilities functions for preprocessing of data.

**User guide:** See the :ref:`preprocessing` section for further details.

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.FeaturesBinarizer
   preprocessing.LongitudinalFeaturesProduct
   preprocessing.LongitudinalFeaturesLagger


.. _api-metrics:

:ref:`metrics`
==============

This module contains some functions to compute some metrics that help evaluate
the performance of learning techniques.

**User guide:** See the :ref:`metrics` section for further details.


.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.support_fdp
   metrics.support_recall
