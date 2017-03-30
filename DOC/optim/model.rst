
Models [`tick.optim.model`]
===========================

Introduction
------------

In `tick` a `Model` class gives information about a statistical model.
Depending on the case, it gives first order information (loss, gradient) or
second order information (hessian norm evaluation).

For now, we have two families of models: Generalized Linear Models and
Hawkes models.

Contents
--------

* :ref:`generalized-linear-models`
* :ref:`model-hawkes`


.. _generalized-linear-models:

Generalized linear models
-------------------------

[Classes from `tick.optim.model`]

Introduction
************

We describe here linear generalized models for supervised learning.
Given training data :math:`(x_i, y_i) \in \mathbb R^d \times \mathbb R`
for :math:`i=1, \ldots, n`, we consider models with a goodness-of fit-that
writes

.. math::
	f(\theta) = \frac 1n \sum_{i=1}^n \ell(y_i, b + x_i^\top \theta),

where :math:`\theta \in \mathbb R^d` is a vector containing coefficients
to be learned, :math:`b \in \mathbb R` is the intercept and
:math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function.
The loss function depends on the model. The following table describes the
different losses implemented in ``tick`` and its associated class :

==========================================  =========================================  ==========================================
Model                                       Loss formula                               Class
==========================================  =========================================  ==========================================
:ref:`model-linreg`                         :math:`\ell(y, z) = \frac 12 (y - z)^2`    `ModelLinReg`
:ref:`model-logreg`                         :math:`\ell(y, z) = \log(1 + \exp(-y z))`  `ModelLogReg`
:ref:`model-poisreg` with exponential link                                             `ModelPoisReg` with ``link="exponential"``
:ref:`model-poisreg` with identity link                                                `ModelPoisReg` with ``link="identity"``
:ref:`coxreg-partial-lik`                                                              `ModelCoxRegPartialLik`
==========================================  =========================================  ==========================================

All model classes allow to compute the loss, the gradient and eventually some
second order informations about the model, see the classes below.
All classes can be used with any solver, by passing them with a solver's ``set_model``
method.

Example
*******
The next example shows how to create a Logistic regression model from
simulated data, and how to use it with a solver (here we use
`AGD`, even if it's certainly not a good choice)::

    from tick.simulation import SimuLogReg, coeffs_sparse_gauss
    from tick.optim.model import ModelLogReg
    from tick.optim.solver import AGD
    from tick.optim.prox import ProxZero

    n_samples = 30000,
    n_features = 100
    # Ground truth coefficients
    coeffs0 = coeffs_sparse_gauss(n_features=n_features, nnz=10)
    # Ground truth intercept
    interc0 = 2
    # Simulation of data: X contains the features matrix and y the labels
    X, y = SimuLogReg(coeffs0, interc=interc0, n_samples=n_samples).simulate()

    # Create a logistic regression model, and pass the data to it
    model = ModelLogReg(fit_intercept=True).fit(X, y)
    # Show a JSON description of the object
    print(model)

    # No penalization is used
    prox = ProxZero()
    # Create a solver, pass to it the model and the penalization
    agd = AGD().set_model(model).set_prox(prox)

    # Launch the solver
    coeffs = agd.solve()


.. _model-hawkes:

Hawkes model
------------

Introduction
************


===================================  ===================================
Model                                Class
===================================  ===================================
:ref:`model-hawkes-exp`              `ModelHawkesFixedExpKernLogLik`
:ref:`model-hawkes-exp-least-sq`     `ModelHawkesFixedExpKernLeastSq`
:ref:`model-hawkes-sumexp-least-sq`  `ModelHawkesFixedSumExpKernLeastSq`
===================================  ===================================


What's under the hood?
----------------------
All model classes have a ``loss`` and ``grad`` method, that are used by batch
algorithms to fit the model. These classes contains a C++ object, that does the
computations. Some methods are hidden within this C++ object, and are accessible
only through C++ (such as ``loss_i`` and ``grad_i`` that compute the gradient
using the single data point :math:`(x_i, y_i)`). These hidden methods are used
in the stochastic solvers, and are available through C++ only for efficiency.
These methods are described in the C++ documentation here
(TODO: add the link to doxygen)
