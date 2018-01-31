
.. _linear_model:

========================
:mod:`tick.linear_model`
========================

This module proposes tools for the inference and simulation of (generalized)
linear models, including among others linear, logistic and Poisson regression,
with a large set of penalization techniques and solvers.
It also proposes hinge losses for supervised learning, namely the hinge,
quadratic hinge and smoothed hinge losses.

1. Introduction
===============

Given training data :math:`(x_i, y_i) \in \mathbb R^d \times \mathbb R`
for :math:`i=1, \ldots, n`, `tick` considers, when solving a generalized linear
model an objective of the form

.. math::
    \frac 1n \sum_{i=1}^n \ell(y_i, x_i^\top w + b) + g(w),

where:

* :math:`w \in \mathbb R^d` is a vector containing the model weights;
* :math:`b \in \mathbb R` is the population intercept;
* :math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function
* :math:`g` is a penalization function, see :ref:`prox` for a list of available penalization technique

Depending on the model the label can be binary :math:`y_i \in \{ -1, 1 \}`
(binary classification), discrete :math:`y_i \in \mathbb N` (Poisson models,
see below) or continuous :math:`y_i \in \mathbb R` (least-squares regression).
The loss function :math:`\ell` depends on the considered model, that are listed
in :ref:`linear_model-model` below. This modules proposes also end-user
learner classes described below.

2. Learners
===========

The following table lists the different learners that allows to train a
model with several choices of penalizations and solvers.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LinearRegression
   linear_model.LogisticRegression
   linear_model.PoissonRegression

These classes follow whenever possible the scikit-learn API, namely ``fit``
and ``predict`` methods, and follow the same naming conventions.

Example
-------
.. plot:: ../examples/plot_logistic_adult.py
    :include-source:


.. _linear_model-model:

3. Models
=========

In ``tick`` a ``model`` class gives information about a statistical model.
Depending on the case, it gives first order information (loss, gradient) or
second order information (hessian norm evaluation).
In this module, a model corresponds to the choice of a loss function, that are
described below. The advantages of using one or another loss are explained in
the documentation of the classes themselves.
The following table lists the different losses implemented for now in this
module, its associated class and label type.

========================================  ==============  ==========  ==========================================
Model                                     Type            Label type  Class
========================================  ==============  ==========  ==========================================
Linear regression                         Regression      Continuous  :class:`ModelLinReg <tick.linear_model.ModelLinReg>`
Logistic regression                       Classification  Binary      :class:`ModelLogReg <tick.linear_model.ModelLogReg>`
Poisson regression (identity link)        Count data      Integer     :class:`ModelPoisReg <tick.linear_model.ModelPoisReg>`
Poisson regression (exponential link)     Count data      Integer     :class:`ModelPoisReg <tick.linear_model.ModelPoisReg>`
Hinge loss                                Classification  Binary      :class:`ModelHinge <tick.linear_model.ModelHinge>`
Quadratic hinge loss                      Classification  Binary      :class:`ModelQuadraticHinge <tick.linear_model.ModelQuadraticHinge>`
Smoothed hinge loss                       Classification  Binary      :class:`ModelSmoothedHinge <tick.linear_model.ModelSmoothedHinge>`
========================================  ==============  ==========  ==========================================

3.1. Description of the available models
----------------------------------------

On the following graph we represent the different losses available in tick for
supervised linear learning. Note that :class:`ModelHuber <tick.robust.ModelHuber>`,
:class:`ModelEpsilonInsensitive <tick.robust.ModelEpsilonInsensitive>`,
:class:`ModelModifiedHuber <tick.robust.ModelModifiedHuber>` and
:class:`ModelAbsoluteRegression <tick.robust.ModelAbsoluteRegression>` are available
through the :ref:`robust` module.


.. plot:: modules/code_samples/plot_linear_model_losses.py


:class:`ModelLinReg <tick.linear_model.ModelLinReg>`
****************************************************
This is least-squares regression with loss

.. math::
    \ell(y, y') = \frac 12 (y - y')^2

for :math:`y, y' \in \mathbb R`

----------------------------------------

:class:`ModelLogReg <tick.linear_model.ModelLogReg>`
****************************************************

Logistic regression for binary classification with loss

.. math::
    \ell(y, y') = \log(1 + \exp(-y y'))

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------

:class:`ModelHinge <tick.linear_model.ModelHinge>`
**************************************************

This is the hinge loss for binary classification given by

.. math::
    \ell(y, y') =
    \begin{cases}
    1 - y y' &\text{ if } y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------

:class:`ModelQuadraticHinge <tick.linear_model.ModelQuadraticHinge>`
********************************************************************

This is the quadratic hinge loss for binary classification given by

.. math::
    \ell(y, y') =
    \begin{cases}
    \frac 12 (1 - y y')^2 &\text{ if } y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------

:class:`ModelSmoothedHinge <tick.linear_model.ModelSmoothedHinge>`
******************************************************************

This is the smoothed hinge loss for binary classification given by

.. math::
    \ell(y, y') =
    \begin{cases}
    1 - y y' - \frac \delta 2 &\text{ if } y y' \leq 1 - \delta \\
    \frac{(1 - y y')^2}{2 \delta} &\text{ if } 1 - \delta < y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`,
where :math:`\delta \in (0, 1)` can be tuned using the ``smoothness`` parameter.
Note that :math:`\delta = 0` corresponds to the hinge loss.

----------------------------------------

:class:`ModelPoisReg <tick.linear_model.ModelPoisReg>`
******************************************************

Poisson regression with exponential link with loss corresponds to the loss

.. math::
    \ell(y, y') = e^{y'} - y y'

for :math:`y \in \mathbb N` and :math:`y' \in \mathbb R` and is obtained
using ``link='exponential'``. Poisson regression with identity link, namely with loss

.. math::
    \ell(y, y') = y' - y \log(y')

for :math:`y \in \mathbb N` and :math:`y' > 0` is obtained using
``link='identity'``.

3.2. The ``model`` class API
----------------------------

All model classes allow to compute the loss (value of the objective function :math:`f`) and
its gradient. Let us illustrate this with the logistic regression model.
First, we need to simulate data.

.. testcode:: [optim-model-glm]

    import numpy as np
    from tick.linear_model import SimuLogReg
    from tick.simulation import weights_sparse_gauss

    n_samples, n_features = 2000, 50
    weights0 = weights_sparse_gauss(n_weights=n_features, nnz=10)
    intercept0 = 1.
    X, y = SimuLogReg(weights0, intercept=intercept0, seed=123,
                      n_samples=n_samples, verbose=False).simulate()

Now, we can create the model object for logistic regression

.. testcode:: [optim-model-glm]

    from tick.linear_model import ModelLogReg

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

.. plot:: modules/code_samples/plot_grad_coeff0.py

We observe that the gradient near the optimum is much smaller than far from it.
Model classes can be used with any solver class, by simply passing them through
the solver's ``set_model`` method, see :ref:`solver`.

.. _simulation-linear-model:

**What's under the hood?**

All model classes have a ``loss`` and ``grad`` method, that are used by batch
algorithms to fit the model. These classes contains a C++ object, that does the
computations. Some methods are hidden within this C++ object, and are accessible
only through C++ (such as ``loss_i`` and ``grad_i`` that compute the gradient
using the single data point :math:`(x_i, y_i)`). These hidden methods are used
in the stochastic solvers, and are available through C++ only for efficiency.

4. Simulation
=============

Simulation of several linear models can be done using the following classes.
All simulation classes simulates a features matrix :math:`\boldsymbol X` with rows :math:`x_i`
and a labels vector :math:`y` with coordinates :math:`y_i` for :math:`i=1, \ldots, n`, that
are i.i.d realizations of a random vector :math:`X` and a scalar random variable :math:`Y`.
The conditional distribution of :math:`Y | X` is :math:`\mathbb P(Y=y | X=x)`,
where :math:`\mathbb P` depends on the considered model.

=====================================  =============================================  ============================
Model                                  Distribution :math:`\mathbb P(Y=y | X=x)`      Class
=====================================  =============================================  ============================
Linear regression                      :math:`\text{Normal}(w^\top x + b, \sigma^2)`  :class:`SimuLinReg <tick.linear_model.SimuLinReg>`
Logistic regression                    :math:`\text{Binomial}(w^\top x + b)`          :class:`SimuLogReg <tick.linear_model.SimuLogReg>`
Poisson regression (identity link)     :math:`\text{Poisson}(w^\top x + b)`           :class:`SimuPoisReg <tick.linear_model.SimuPoisReg>` with ``link="identity"``
Poisson regression (exponential link)  :math:`\text{Poisson}(e^{w^\top x + b})`       :class:`SimuPoisReg <tick.linear_model.SimuPoisReg>` with ``link="exponential"``
=====================================  =============================================  ============================

**Example**

.. plot:: ../examples/plot_simulation_linear_model.py
    :include-source:
