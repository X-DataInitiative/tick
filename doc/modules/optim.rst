
.. _optim:

=======================================
:mod:`tick.optim`: optimization toolbox
=======================================

This is the core module of ``tick`` : an optimization toolbox, that allows
to combine models (``model`` classes), penalizations (``prox`` classes) and
solvers (``solver`` classes) in many ways.
Most of the optimization problems considered in ``tick`` (but not all)
can be written as

.. math::
    \min_w f(w) + g(w)

where :math:`f` is a goodness-of-fit term and :math:`g` is a function penalizing :math:`w`.
Depending on the problem, you might want to use a specific algorithm to solve it.
This optimization module is therefore organized in the following three main submodules.

Contents
========

========================  ====================================  ============
Module API                Documentation        Description
========================  ====================================  ============
:mod:`tick.optim.model`   :ref:`Model classes <optim-model>`    Gives information about :math:`f`: value, gradient, hessian
:mod:`tick.optim.prox`    :ref:`Prox classes <optim-prox>`      Gives information about :math:`g`: value and proximal operator
:mod:`tick.optim.solver`  :ref:`Solver classes <optim-solver>`  Given a model and a prox, minimizes :math:`f + g`
========================  ====================================  ============


.. _optim-first-example:

A First example
===============

Here is an example of combination of a ``model`` a ``prox`` and a ``solver`` to
compare the training time of several solvers for logistic regression with the
elastic-net penalization.
Note that, we specify a ``range=(0, n_features)`` so that the intercept is not penalized
(see :ref:`Prox classes <optim-prox>` below for more details).

.. plot:: modules/code_samples/optim/plot_optim_comparison.py
    :include-source:


.. _optim-model:

1. :mod:`tick.optim.model`: model classes
=========================================

In ``tick`` a ``model`` class gives information about a statistical model.
Depending on the case, it gives first order information (loss, gradient) or
second order information (hessian norm evaluation).

1.1. The ``model`` class API
----------------------------

All model classes allow to compute the loss (value of the objective function :math:`f`) and
its gradient. Let us illustrate this with the logistic regression model. First, we simulate
some data, see :ref:`linear model simulation <simulation-linear-model>` to have more information about this.

.. testcode:: [optim-model-glm]

    import numpy as np
    from tick.simulation import SimuLogReg, weights_sparse_gauss

    n_samples, n_features = 2000, 50
    weights0 = weights_sparse_gauss(n_weights=n_features, nnz=10)
    intercept0 = 1.
    X, y = SimuLogReg(weights0, intercept=intercept0, seed=123,
                      n_samples=n_samples, verbose=False).simulate()

Now, we can create the model object for logistic regression


.. testcode:: [optim-model-glm]

    from tick.optim.model import ModelLogReg

    model = ModelLogReg(fit_intercept=True).fit(X, y)
    print(model)

outputs

.. testoutput:: [optim-model-glm]
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    {
      "fit_intercept": true,
      "n_calls_grad": 0,
      "n_calls_loss": 0,
      "n_calls_loss_and_grad": 0,
      "n_coeffs": 51,
      "n_features": 50,
      "n_passes_over_data": 0,
      "n_samples": 2000,
      "n_threads": 1,
      "name": "ModelLogReg"
    }

Printing any object in tick returns a json formatted description of it.
We see that this model uses 50 features, 51 coefficients (including the intercept),
and that it received 2000 sample points. Now we can compute the loss of the model using
the ``loss`` method (its objective, namely the value of the function :math:`f`
to be minimized) by using

.. testcode:: [optim-model-glm]

    coeffs0 = np.concatenate([weights0, [intercept0]])
    print(model.loss(coeffs0))

which outputs

.. testoutput:: [optim-model-glm]
    :hide:

    ...

.. code-block:: python

    0.3551082120992899

while

.. testcode:: [optim-model-glm]

    print(model.loss(np.ones(model.n_coeffs)))

outputs

.. testoutput:: [optim-model-glm]
    :hide:

    ...

.. code-block:: python

    5.793300908869233

which is explained by the fact that the loss is larger for a parameter which is far from
the ones used for the simulation.
The gradient of the model can be computed using the ``grad`` method

.. code-block:: python

    _, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 3))
    ax[0].stem(model.grad(coeffs0))
    ax[0].set_title(r"$\nabla f(\mathrm{coeffs0})$", fontsize=16)
    ax[1].stem(model.grad(np.ones(model.n_coeffs)))
    ax[1].set_title(r"$\nabla f(\mathrm{coeffs1})$", fontsize=16)

which plots

.. plot:: modules/code_samples/optim/plot_grad_coeff0.py

We observe that the gradient near the optimum is much smaller than far from it.

Model classes can be used with any solver class, by simply passing them using the
solver's ``set_model`` method, see the :ref:`example given above <optim-first-example>`.


.. _optim-model-glm:

1.2. Generalized linear models
------------------------------

We describe here generalized linear models for supervised learning.
Given training data :math:`(x_i, y_i) \in \mathbb R^d \times \mathbb R`
for :math:`i=1, \ldots, n`, we consider models with a goodness-of-fit that
writes

.. math::
	f(w, b) = \frac 1n \sum_{i=1}^n \ell(y_i, b + x_i^\top w),

where :math:`w \in \mathbb R^d` is a vector containing the model weights,
:math:`b \in \mathbb R` is the intercept and
:math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function.
The loss function depends on the model. The following table describes the
different losses implemented for now in tick and its associated class.

========================================  ===========================================  ==========================================
Model                                      Loss formula                                Class
========================================  ===========================================  ==========================================
Linear regression                         :math:`\ell(y, y') = \frac 12 (y - y')^2`    :class:`ModelLinReg <tick.optim.model.ModelLinReg>`
Logistic regression                       :math:`\ell(y, y') = \log(1 + \exp(-y y'))`  :class:`ModelLogReg <tick.optim.model.ModelLogReg>`
Poisson regression with exponential link  :math:`\ell(y, y') = y' - y \log(y')`        :class:`ModelPoisReg <tick.optim.model.ModelPoisReg>` with ``link="exponential"``
Poisson regression with identity link     :math:`\ell(y, y') = e^{y'} - y y'`          :class:`ModelPoisReg <tick.optim.model.ModelPoisReg>` with ``link="identity"``
========================================  ===========================================  ==========================================


1.3 Generalized linear models with individual intercepts
--------------------------------------------------------

The setting is the same as with generalized linear models, but where we used an
individual intercept :math:`b_i` for each :math:`i=1, \ldots, n`.
Namely we consider a goodness-of-fit of the form

.. math::

    f(w, b) = \frac 1n \sum_{i=1}^n \ell(y_i, b_i + x_i^\top w),

where :math:`w \in \mathbb R^d` is a vector containing the model weights,
:math:`b \in \mathbb R^n` is a vector of individual intercepts and
:math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function.
Estimation of :math:`b` under a sparse penalization (such as L1 or
Sorted L1, see :ref:`prox classes <optim-prox>` below) allows to detect outliers
using this model.


=================================  =========================================  ==========================================
Model                              Loss formula                               Class
=================================  =========================================  ==========================================
Linear regression with intercepts  :math:`\ell(y, y') = \frac 12 (y - y')^2`  :class:`ModelLinRegWithIntercepts <tick.optim.model.ModelLinRegWithIntercepts>`
=================================  =========================================  ==========================================


.. _optim-model-survival:

1.4. Survival analysis
----------------------

.. todo::
    Quick survival analysis presentation here?

.. todo::

    Describe Cox model

.. todo::
    Describe Self Control Case Series model

=================================  ==============================
Model                              Class
=================================  ==============================
Cox regression partial likelihood  :class:`ModelCoxRegPartialLik <tick.optim.model.ModelCoxRegPartialLik>`
Self Control Case Series           :class:`ModelSCCS <tick.optim.model.ModelSCCS>`
=================================  ==============================


.. _optim-model-hawkes:

1.5. Hawkes models
------------------

Hawkes processes are point processes defined by the intensities:

.. math::
    \forall i \in [1 \dots D], \quad
    \lambda_i(t) = \mu_i + \sum_{j=1}^D \int \phi_{ij}(t - s) dN_j(s)

where

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels
* :math:`dN_j` are the processes differentiates

One way to infer Hawkes processes is to suppose their kernels have a
parametric shape. Usually kernels have an exponential parametrization as it
allows very fast computations.

In *tick*, three exponential models are implemented. They differ by the
parametrization of the kernel (exponential or sum-exponential) or by the loss
function used (least squares or log-likelihood).

==============================================================  ===============================
Model                                                           Class
==============================================================  ===============================
Least-squares for Hawkes model with exponential kernels         :class:`ModelHawkesFixedExpKernLeastSq <tick.optim.model.ModelHawkesFixedExpKernLeastSq>`
Log-likelihood for Hawkes model with exponential kernels        :class:`ModelHawkesFixedExpKernLogLik <tick.optim.model.ModelHawkesFixedExpKernLogLik>`
Least-squares for Hawkes model with sum of exponential kernels  :class:`ModelHawkesFixedSumExpKernLeastSq <tick.optim.model.ModelHawkesFixedSumExpKernLeastSq>`
==============================================================  ===============================


.. _optim-prox:

2. :mod:`tick.optim.prox`: proximal operators
=============================================

This module provides several proximal operators for the regularization of the weights
of a model. The proximal operator of a convex function :math:`g`
at some point :math:`w` is defined as the unique minimizer of the problem

.. math::
   \text{prox}_{g}(w, t) = \text{argmin}_{w'} \Big\{ \frac 12 \| w - w' \|_2^2 + t g(w') \Big\}

where :math:`t > 0` is a regularization parameter and :math:`\| \cdot \|_2` is the
Euclidean norm. Note that in the particular case where :math:`g(w) = \delta_{C}(w)`,
with :math:`C` a convex set, then :math:`\text{prox}_g` is a projection
operator (here :math:`\delta_{C}(w) = 0` if :math:`w \in C`
and :math:`+\infty` otherwise).

Note that depending on the problem, :math:`g` might actually be used only a subset of
entries of :math:`w`.
For instance, for generalized linear models, :math:`w` contains the model weights and
an intercept, which is not penalized, see :ref:`generalized linear models <optim-model-glm>`.
Indeed, in all ``prox`` classes, an optional ``range`` parameter is available, to apply
the regularization only to a subset of entries of :math:`w`.

2.1 The ``prox`` class API
--------------------------

Let us describe the ``prox`` API with the :class:`ProxL1<tick.optim.prox.ProxL1>`
class, that provides the proximal operator of the function :math:`g(w) = s \|w\|_1 = s \sum_{j=1}^d |w_j|`.


.. testcode:: [optim-model-prox]

    import numpy as np
    from tick.optim.prox import ProxL1

    prox = ProxL1(strength=1e-2)
    print(prox)

prints

.. testoutput:: [optim-model-prox]

    {
      "name": "ProxL1",
      "positive": false,
      "range": null,
      "strength": 0.01
    }

The ``positive`` parameter allows to enforce positivity, namely when ``positive=True`` then
the considered function is actually :math:`g(w) = s \|w\|_1 + \delta_{C}(x)` where :math:`C` is
the set of vectors with non-negative coordinates.
Note that no ``range`` was specified to this prox so that it is null (``None``) for now.


.. testcode:: [optim-model-prox]

    prox = ProxL1(strength=1e-2, range=(0, 30), positive=True)
    print(prox)

prints

.. testoutput:: [optim-model-prox]

    {
      "name": "ProxL1",
      "positive": true,
      "range": [
        0,
        30
      ],
      "strength": 0.01
    }

The parameter :math:`s` corresponds to the strength of penalization, and can be tuned using
the ``strength`` parameter.

All ``prox`` classes provide a method ``call`` that computes :math:`\text{prox}_{g}(w, t)`
where :math:`t` is a parameter passed using the ``step`` argument.
The output of ``call`` can optionally be passed using the ``out`` argument (this avoid unnecessary copies, and
thus extra memory allocation).

.. plot:: modules/code_samples/optim/plot_prox_api.py
    :include-source:

The value of :math:`g` is simply obtained using the ``value`` method

.. testcode:: [optim-model-prox]

    prox = ProxL1(strength=1., range=(5, 10))
    val = prox.value(np.arange(10, dtype=np.double))
    print(val)

simply prints

.. testoutput:: [optim-model-prox]

    35.0

which corresponds to the sum of integers between 5 and 9 included.


2.2 Available operators
-----------------------

The list of available operators in ``tick`` given in the next table.

=======================  ========================================================================================  ==============
Penalization             Function                                                                                  Class
=======================  ========================================================================================  ==============
Identity                 :math:`g(w) = 0`                                                                          :class:`ProxZero <tick.optim.prox.ProxZero>`
Non-negative constraint  :math:`g(w) = s \delta_C(w)` where :math:`C=` set of vectors with non-negative entries    :class:`ProxPositive <tick.optim.prox.ProxPositive>`
Equality constraint      :math:`g(w) = s \delta_C(w)` where :math:`C=` set of vectors with identical entries       :class:`ProxEquality <tick.optim.prox.Equality>`
L1 norm                  :math:`g(w) = s \sum_{j=1}^d |w_j|`                                                       :class:`ProxL1 <tick.optim.prox.ProxL1>`
L1 norm with weights     :math:`g(w) = s \sum_{j=1}^d c_j |w_j|`                                                   :class:`ProxL1w <tick.optim.prox.ProxL1w>`
Ridge                    :math:`g(w) = s \sum_{j=1}^d \frac{w_j^2}{2}`                                             :class:`ProxL2Sq <tick.optim.prox.ProxL2Sq>`
Elastic-net              :math:`g(w) = s \Big(\sum_{j=1}^{d} \alpha |w_j| + (1 - \alpha) \frac{w_j^2}{2} \Big)`    :class:`ProxElasticNet <tick.optim.prox.ProxElasticNet>`
Total-variation          :math:`g(w) = s \sum_{j=2}^d |w_j - w_{j-1}|`                                             :class:`ProxTV <tick.optim.prox.ProxTV>`
Nuclear norm             :math:`g(w) = s \sum_{j=1}^{q} \sigma_j(w)`                                               :class:`ProxNuclear <tick.optim.prox.ProxNuclear>`
Sorted L1                :math:`g(w) = s \sum_{j=1}^{d} c_j |w_{(j)}|` where :math:`|w_{(j)}|` is decreasing       :class:`ProxSlope <tick.optim.prox.ProxSlope>`
=======================  ========================================================================================  ==============

Another ``prox`` class is the :class:`ProxMulti <tick.optim.prox.ProxMulti>` that allows
to combine any proximal operators together.
It simply applies sequentially each operator passed to :class:`ProxMulti <tick.optim.prox.ProxMulti>`,
one after the other. Here is an example of combination of a total-variation penalization and L1 penalization
applied to different parts of a vector.

.. plot:: modules/code_samples/optim/plot_prox_multi.py
    :include-source:

Example
-------
Here is an illustration of the effect of these proximal operators on an example.

.. plot:: ../examples/plot_prox_example.py
    :include-source:


.. _optim-solver:

3. :mod:`tick.optim.solver`: solvers
====================================

This module contains all the solvers available in ``tick``.
It features two types of solvers: deterministic and stochastic.
Deterministic solvers use a full pass over
data at each iteration, while stochastic solvers make ``epoch_size`` iterations
within each iteration.

3.1 The ``solver`` class API
----------------------------

All the solvers have a ``set_model`` method to pass the model to be trained, and
a ``set_prox`` method to pass the penalization.
The solver is launched using the ``solve`` method to which a starting point and
eventually a step-size can be given. Here is an example

.. testcode::

    import numpy as np
    from tick.simulation import SimuLogReg, weights_sparse_gauss
    from tick.optim.solver import SVRG
    from tick.optim.model import ModelLogReg
    from tick.optim.prox import ProxElasticNet

    n_samples, n_features = 5000, 10
    weights0 = weights_sparse_gauss(n_weights=n_features, nnz=3)
    intercept0 = 1.
    X, y = SimuLogReg(weights0, intercept=intercept0, seed=123,
                      n_samples=n_samples, verbose=False).simulate()

    model = ModelLogReg(fit_intercept=True).fit(X, y)
    prox = ProxElasticNet(strength=1e-3, ratio=0.5, range=(0, n_features))

    svrg = SVRG(tol=0., max_iter=5, print_every=1).set_model(model).set_prox(prox)
    x0 = np.zeros(model.n_coeffs)
    minimizer = svrg.solve(x0, step=1 / model.get_lip_max())
    print("\nfound minimizer\n", minimizer)

which outputs

.. testoutput::
    :hide:
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Launching the solver SVRG...
      n_iter  |    obj    |  rel_obj
            0 |    ...    |    ...
            1 |    ...    |    ...
            2 |    ...    |    ...
            3 |    ...    |    ...
            4 |    ...    |    ...
            5 |    ...    |    ...
    Done solving using SVRG in ... seconds

    found minimizer
     [...  ...  ...  ...  ... ... ... ... ... ... ...]

.. code-block:: none

    Launching the solver SVRG...
      n_iter  |    obj    |  rel_obj
            0 |  5.29e-01 |  2.37e-01
            1 |  5.01e-01 |  5.15e-02
            2 |  4.97e-01 |  8.44e-03
            3 |  4.97e-01 |  5.00e-04
            4 |  4.97e-01 |  2.82e-05
            5 |  4.97e-01 |  7.28e-07

    Done solving using SVRG in 0.03281998634338379 seconds

    found minimizer
     [ 0.01992683  0.00456966 -0.16595686 -0.08619878  0.01059461  0.6144692
      0.0049031  -0.07767023  0.07550217  1.18493663  0.9424508 ]

Note the argument ``step=1 / model.get_lip_max())`` passed to the ``solve`` method that gives
an automatic tuning of the step size.


3.2 Available solvers
---------------------

Here is the list of the solvers available in ``tick``.

=======================================================  ========================================
Solver                                                   Class
=======================================================  ========================================
Proximal gradient descent                                :class:`GD <tick.optim.solver.GD>`
Accelerated proximal gradient descent                    :class:`AGD <tick.optim.solver.AGD>`
Broyden, Fletcher, Goldfarb, and Shannon (quasi-newton)  :class:`BFGS <tick.optim.solver.BFGS>`
Self-Concordant Proximal Gradient Descent                :class:`SCPG <tick.optim.solver.SCPG>`
Stochastic Gradient Descent                              :class:`SGD <tick.optim.solver.SGD>`
Adaptive Gradient Descent solver                         :class:`AdaGrad <tick.optim.solver.AdaGrad>`
Stochastic Variance Reduced Descent                      :class:`SVRG <tick.optim.solver.SVRG>`
Stochastic Dual Coordinate Ascent                        :class:`SDCA <tick.optim.solver.SDCA>`
=======================================================  ========================================


4. What's under the hood?
=========================

All model classes have a ``loss`` and ``grad`` method, that are used by batch
algorithms to fit the model. These classes contains a C++ object, that does the
computations. Some methods are hidden within this C++ object, and are accessible
only through C++ (such as ``loss_i`` and ``grad_i`` that compute the gradient
using the single data point :math:`(x_i, y_i)`). These hidden methods are used
in the stochastic solvers, and are available through C++ only for efficiency.
