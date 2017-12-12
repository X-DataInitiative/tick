:orphan:

.. _api:

=============
API Reference
=============

This is the full class and function references of tick. Please look at
the modules documentation cited below for more examples and use cases,
since direct class and function API is not enough for understanding their uses.

.. _api-inference:

:mod:`tick.inference`: Inference classes
========================================

This module contains all classes giving inference tools, intended for end-users.

**User guide:** See the :ref:`inference` section for further details.

.. automodule:: tick.inference
   :no-members:
   :no-inherited-members:

Generalized linear models
-------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.LinearRegression
   inference.LogisticRegression
   inference.PoissonRegression

Robust Analysis
---------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   inference.std_mad
   inference.std_iqr

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.RobustLinearRegression


Survival Analysis
-----------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   inference.nelson_aalen
   inference.kaplan_meier

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.CoxRegression

Hawkes
------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.HawkesExpKern
   inference.HawkesSumExpKern
   inference.HawkesEM
   inference.HawkesADM4
   inference.HawkesBasisKernels
   inference.HawkesSumGaussians
   inference.HawkesConditionalLaw

.. _api-optim-model:

:mod:`tick.optim.model`: Models classes
=======================================

This module contains classes giving computational informations about the models available
in tick.

**User guide:** See the :ref:`optim-model` section for further details.

.. automodule:: tick.optim.model
   :no-members:
   :no-inherited-members:


Linear models for regression
----------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelLinReg
   optim.model.ModelHuber
   optim.model.ModelAbsoluteRegression
   optim.model.ModelEpsilonInsensitive


Linear models for binary classification
---------------------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelLogReg
   optim.model.ModelHinge
   optim.model.ModelSmoothedHinge
   optim.model.ModelQuadraticHinge
   optim.model.ModelModifiedHuber


Linear models for count data
----------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelPoisReg


Linear models with individual intercepts (outliers detection)
-------------------------------------------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelLinRegWithIntercepts


Survival analysis
-----------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelCoxRegPartialLik
   optim.model.ModelSCCS


Hawkes
------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelHawkesFixedExpKernLogLik
   optim.model.ModelHawkesFixedExpKernLeastSq
   optim.model.ModelHawkesFixedSumExpKernLogLik
   optim.model.ModelHawkesFixedSumExpKernLeastSq


.. _api-prox:

:mod:`tick.prox`: Proximal operators classes
==================================================

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

:mod:`tick.solver`: Solver classes
========================================

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


.. _api-plot:

:mod:`tick.plot`: Plotting utilities
====================================

This module contains some utilities functions for plotting

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


.. _api-preprocessing:

:mod:`tick.preprocessing`: Preprocessing utilities
==================================================

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

:mod:`tick.metrics`: Metrics utilities
======================================

This module contains some functions to compute some metrics that help evaluate
the performance of learning techniques.

Functions
---------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.support_fdp
   metrics.support_recall


.. _api-simulation:

:mod:`tick.simulation`: Simulation classes and functions
========================================================

This module contains all simulation tools available in tick.

**User guide:** See the :ref:`simulation` section for further details.

Generalized linear models
-------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuLinReg
   simulation.SimuLogReg
   simulation.SimuPoisReg

Point processes
---------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuCoxReg
   simulation.SimuPoissonProcess
   simulation.SimuInhomogeneousPoisson
   simulation.SimuHawkes
   simulation.SimuHawkesExpKernels
   simulation.SimuHawkesSumExpKernels
   simulation.SimuHawkesMulti

Hawkes kernels
--------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.HawkesKernelExp
   simulation.HawkesKernelSumExp
   simulation.HawkesKernelPowerLaw
   simulation.HawkesKernelTimeFunc
   base.TimeFunction

Features generators
-------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz
   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss


.. _api-datasets:

:mod:`tick.dataset`: Real world dataset
=======================================

**User guide:** See the :ref:`dataset` section for further details.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.fetch_tick_dataset
   dataset.fetch_hawkes_bund_data
