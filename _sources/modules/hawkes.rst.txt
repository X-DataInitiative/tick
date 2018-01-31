
.. _hawkes:

==================
:mod:`tick.hawkes`
==================

This module proposes a comprehensive set of tools for the inference and the
simulation of Hawkes processes, with both parametric and non-parametric
estimation techniques and flexible tools for simulation.

**Contents**

* :ref:`hawkes-intro`
* :ref:`hawkes-learners`
* :ref:`hawkes-simulation`
* :ref:`hawkes-models`

.. _hawkes-intro:

1. Introduction
===============

A Hawkes process is a :math:`D`-dimensional counting process
:math:`N(t) = (N_1(t) \cdots N_D(t))`, where each coordinate is a counting
process :math:`N_i(t) = \sum_{k \geq 1} \mathbf 1_{t_{i, k} \leq t}`, with
:math:`t_{i, 1}, t_{i, 2}, \ldots` being the ticks or timestamps observed on
component :math:`i`. The intensity of :math:`N` is given by

.. math::

    \lambda_i(t) = \mu_i + \sum_{j=1}^D \sum_{k \ : \ t_{j, k} < t} \phi_{ij}(t - t_{j, k})

for :math:`i = 1, \ldots, D`. Such an intensity induces a cluster effect, namely activity one a node
:math:`j` induces intensity on another node :math:`i`, with an impact encoded
by the *kernel* function :math:`\phi_{ij}`. In the the above formula, we have

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels.

Note that different choices for the shape of the kernels correspond to different
models, described below, and two strategies for parametric inference are available,
either least-squares estimation or maximum likelihood estimation, both with several
choices of penalization.

.. _hawkes-learners:

2. Learners
===========

This module proposes learners, that are meant to be very user friendly
and are most of the time good enough to infer Hawkes processes based on data,
with both parametric and non-parametric approaches.

2.1. Parametric Hawkes learners
-------------------------------

One way to infer Hawkes processes is to suppose that their kernels have a
parametric shape. A standard assumption is an exponential parametrization since it
allows very fast computations, thanks to tricky recurrence formulas.
The following learners propose in their parameters several choices of
penalization and solvers.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesExpKern
   HawkesSumExpKern
   HawkesADM4
   HawkesSumGaussians

Example
*******

.. plot:: modules/code_samples/plot_hawkes_sum_exp_kernels.py
    :include-source:

2.2. Non-parametric Hawkes learners
-----------------------------------

The following Hawkes learners perform non-parametric estimation of the kernels
and hence don't rely on the previous exponential parametrization.
However, this methods are less scalable with respect to the number of nodes :math:`D`.
These learners might be used to infer more exotic kernels.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesEM
   HawkesBasisKernels
   HawkesConditionalLaw
   HawkesCumulantMatching

Example
*******

.. plot:: ../examples/plot_hawkes_em.py
    :include-source:


.. _hawkes-simulation:

3. Point process simulation
===========================

In this section with describe how to simulate point processes and Hawkes
processes using `tick`. First, we need to describe two tools giving flexibility
in the simulation of point processes : time functions and kernels for Hawkes
process simulation.

3.1. Time function
------------------

A ``TimeFunction`` is a class allowing to define a function on
:math:`[0, \infty)`. It uses several types of interpolation to determine its value
between two points. It is used for the simulation of an inhomogeneous Poisson
process and some Hawkes processes.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

    base.TimeFunction

Example
*******

.. plot:: modules/code_samples/plot_time_function.py
    :include-source:

3.2. Kernels for Hawkes process simulation
------------------------------------------

A Hawkes process is defined through its kernels which are functions defined on
:math:`[0, \infty)`. The following kernels are available for simulation.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesKernel0
   HawkesKernelExp
   HawkesKernelSumExp
   HawkesKernelPowerLaw
   HawkesKernelTimeFunc

Example
*******

.. plot:: modules/code_samples/plot_hawkes_kernels.py
    :include-source:


3.3. Poisson processes
----------------------

Both homogeneous and inhomogeneous Poisson process might be simulated with
`tick` thanks to the following classes.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuPoissonProcess
   SimuInhomogeneousPoisson


Examples
********

A Poisson process with constant intensity

.. plot:: modules/code_samples/plot_poisson_constant_intensity.py
    :include-source:

A Poisson process with variable intensity. In this case, the intensity is
defined through a `tick.base.TimeFunction`

.. plot:: ../examples/plot_poisson_inhomogeneous.py
    :include-source:

3.4. Hawkes processes
---------------------

Simulation of Hawkes processes can be done using the following classes. The
main class :class:`SimuHawkes <tick.hawkes.SimuHawkes>` might use any type of
kernels and will perform simulation. For some specific cases there are some classes
dedicated to a type of kernel: exponential or sum of exponential kernels.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuHawkes
   SimuHawkesExpKernels
   SimuHawkesSumExpKernels

.. plot:: modules/code_samples/plot_hawkes_1d_simu.py
    :include-source:

.. plot:: modules/code_samples/plot_hawkes_multidim_simu.py
    :include-source:

.. _hawkes-models:


4. Models for inference
=======================

The following models can be used with a solver to implement cases potentially
not covered in :ref:`hawkes-learners`. Note that these models are used internally
in the learners and feature least-squares and log-likelihood goodness-of-fit for
parametric Hawkes models with an exponential kernel or a sum of exponential
kernels.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHawkesExpKernLeastSq
   ModelHawkesExpKernLogLik
   ModelHawkesSumExpKernLeastSq
   ModelHawkesSumExpKernLogLik
