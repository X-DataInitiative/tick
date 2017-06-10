

.. _inference:

====================================================
:mod:`tick.inference`: user-friendly inference tools
====================================================

These classes are called learners. They are meant to be very user friendly
and are most of the time good enough to infer many models.

These classes aim to be scikit-learn compatible and hence implement a `fit`
method.

.. contents::
    :depth: 3
    :backlinks: none

.. _inference-glm:

1. Generalized linear models
----------------------------

The first learner types concern linear models presented in

:ref:`optim-model-glm`

Theses learners are essentially a combination of

* :ref:`optim-model`
* :ref:`optim-prox`
* :ref:`optim-solver`

which are described in depth in :ref:`optim`.

Classes
^^^^^^^

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.LogisticRegression

Example
^^^^^^^

These learners can be easily applied on real world datasets

.. plot:: ../examples/plot_logistic_adult.py
    :include-source:


.. _inference-hawkes:

2. Hawkes
---------

*tick* also provides learners to infer Hawkes processes.

Hawkes processes are point processes defined by the intensities:

.. math::
    \forall i \in [1 \dots D], \quad
    \lambda_i(t) = \mu_i + \sum_{j=1}^D \int \phi_{ij}(t - s) dN_j(s)

where

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels
* :math:`dN_j` are the processes differentiates

Parametric Hawkes learners
^^^^^^^^^^^^^^^^^^^^^^^^^^

One way to infer Hawkes processes is to suppose their kernels have a
parametric shape. Usually people induces an exponential parametrization as it
allows very fast computations. The models associated to these learners are
presented in

:ref:`optim-model-hawkes`

As for linear models, `tick.inference.HawkesExpKern` and
`tick.inference.HawkesSumExpKern` are combination of solver, model and prox.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.HawkesExpKern
   inference.HawkesSumExpKern
   inference.HawkesADM4
   inference.HawkesSumGaussians

.. plot:: modules/code_samples/inference/plot_hawkes_sum_exp_kernels.py
    :include-source:

Non-parametric Hawkes learners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some other Hawkes learners perform non parametric evaluation of the kernels
and hence don't rely the previous exponential parametrization.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inference.HawkesEM
   inference.HawkesBasisKernels
   inference.HawkesConditionalLaw

These learners might then infer much more exotic kernels

.. plot:: ../examples/plot_hawkes_em.py
    :include-source:


3. Survival analysis
--------------------
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
