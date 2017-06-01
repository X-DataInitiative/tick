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

   inference.LogisticRegression

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

Generalized Linear Models
-------------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelLinReg
   optim.model.ModelLinRegWithIntercepts
   optim.model.ModelLogReg
   optim.model.ModelPoisReg
   optim.model.ModelCoxRegPartialLik

Hawkes
------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.model.ModelHawkesFixedExpKernLogLik
   optim.model.ModelHawkesFixedExpKernLeastSq
   optim.model.ModelHawkesFixedSumExpKernLeastSq


:mod:`tick.optim.prox`: Proximal operators classes
==================================================

This module contains all the proximal operators available in tick.

**User guide:** See the :ref:`optim-prox` section for further details.

.. automodule:: tick.optim.prox
   :no-members:
   :no-inherited-members:

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.prox.ProxZero
   optim.prox.ProxL1
   optim.prox.ProxL1w
   optim.prox.ProxElasticNet
   optim.prox.ProxL2Sq
   optim.prox.ProxMulti
   optim.prox.ProxNuclear
   optim.prox.ProxPositive
   optim.prox.ProxEquality
   optim.prox.ProxSlope
   optim.prox.ProxTV


.. _api-optim-solver:

:mod:`tick.optim.solver`: Solver classes
========================================

This module contains all the solvers available in tick.

**User guide:** See the :ref:`optim-solver` section for further details.

.. automodule:: tick.optim.solver
   :no-members:
   :no-inherited-members:

Batch solvers
-------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.solver.GD
   optim.solver.AGD
   optim.solver.BFGS
   optim.solver.GFB

Stochastic solvers
------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.solver.SGD
   optim.solver.SVRG
   optim.solver.SDCA
   optim.solver.AdaGrad

History
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   optim.history.History


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
   plot.stem
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
