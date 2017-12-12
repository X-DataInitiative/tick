
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
========================  ====================================  ============

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
solver's ``set_model`` method, see :ref:`solver`.


.. _linear-models:

1.2. Linear models
------------------

We describe here (generalized) linear methods for supervised learning.
Given training data :math:`(x_i, y_i) \in \mathbb R^d \times \mathbb R`
for :math:`i=1, \ldots, n`, we consider goodness-of-fit that writes

.. math::
	f(w, b) = \frac 1n \sum_{i=1}^n \ell(y_i, b + x_i^\top w),

where :math:`w \in \mathbb R^d` is a vector containing the model weights,
:math:`b \in \mathbb R` is the intercept and
:math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function.
Note that for binary regression we have actually binary labels :math:`y_i \in \{ -1, 1 \}`
while for counts data (Poisson models, see below) we have natural integer :math:`y_i \in \mathbb N`.

The loss function depends on the model. The advantages of using one or another
are explained in the documentation of the classes themselves.
The following table lists the different losses implemented for now in `tick`,
its associated class and label type.

========================================  ==============  ==========  ==========================================
Model                                     Type            Label type  Class
========================================  ==============  ==========  ==========================================
Linear regression                         Regression      Continuous  :class:`ModelLinReg <tick.optim.model.ModelLinReg>`
Huber regression                          Regression      Continuous  :class:`ModelHuber <tick.optim.model.ModelHuber>`
Epsilon-insensitive regression            Regression      Continuous  :class:`ModelEpsilonInsensitive <tick.optim.model.ModelEpsilonInsensitive>`
Absolute regression                       Regression      Continuous  :class:`ModelAbsoluteRegression <tick.optim.model.ModelAbsoluteRegression>`
Logistic regression                       Classification  Binary      :class:`ModelLogReg <tick.optim.model.ModelLogReg>`
Hinge loss                                Classification  Binary      :class:`ModelHinge <tick.optim.model.ModelHinge>`
Quadratic hinge loss                      Classification  Binary      :class:`ModelQuadraticHinge <tick.optim.model.ModelQuadraticHinge>`
Smoothed hinge loss                       Classification  Binary      :class:`ModelSmoothedHinge <tick.optim.model.ModelSmoothedHinge>`
Modified Huber loss                       Classification  Binary      :class:`ModelModifiedHuber <tick.optim.model.ModelModifiedHuber>`
Poisson regression (identity link)        Count data      Integer     :class:`ModelPoisReg <tick.optim.model.ModelPoisReg>`
Poisson regression (exponential link)     Count data      Integer     :class:`ModelPoisReg <tick.optim.model.ModelPoisReg>`
========================================  ==============  ==========  ==========================================


Regression models
-----------------

.. plot:: modules/code_samples/optim/plot_losses_regression.py


:class:`ModelLinReg <tick.optim.model.ModelLinReg>`
***************************************************
This is least-squares regression with loss

.. math::
    \ell(y, y') = \frac 12 (y - y')^2

for :math:`y, y' \in \mathbb R`

----------------------------------------

:class:`ModelHuber <tick.optim.model.ModelHuber>`
*************************************************

The Huber loss for robust regression (less sensitive to
outliers) is given by

.. math::
    \ell(y, y') =
    \begin{cases}
    \frac 12 (y' - y)^2 &\text{ if } |y' - y| \leq \delta \\
    \delta (|y' - y| - \frac 12 \delta) &\text{ if } |y' - y| > \delta
    \end{cases}

for :math:`y, y' \in \mathbb R`, where :math:`\delta > 0` can be tuned
using the ``threshold`` argument.

----------------------------------------

:class:`ModelEpsilonInsensitive <tick.optim.model.ModelEpsilonInsensitive>`
***************************************************************************

Epsilon-insensitive loss, given by

.. math::
    \ell(y, y') =
    \begin{cases}
    |y' - y| - \epsilon &\text{ if } |y' - y| > \epsilon \\
    0 &\text{ if } |y' - y| \leq \epsilon
    \end{cases}

for :math:`y, y' \in \mathbb R`, where :math:`\epsilon > 0` can be tuned using
the ``threshold`` argument.

----------------------------------------

:class:`ModelAbsoluteRegression <tick.optim.model.ModelAbsoluteRegression>`
***************************************************************************

The L1 loss given by

.. math::
    \ell(y, y') = |y' - y|

for :math:`y, y' \in \mathbb R`

----------------------------------------


Classification models
---------------------

.. plot:: modules/code_samples/optim/plot_losses_classification.py


:class:`ModelLogReg <tick.optim.model.ModelLogReg>`
***************************************************
Logistic regression for binary classification with loss

.. math::
    \ell(y, y') = \log(1 + \exp(-y y'))

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------

:class:`ModelHinge <tick.optim.model.ModelHinge>`
*************************************************

This is the hinge loss given by

.. math::
    \ell(y, y') =
    \begin{cases}
    1 - y y' &\text{ if } y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------


:class:`ModelQuadraticHinge <tick.optim.model.ModelQuadraticHinge>`
*******************************************************************

This is the quadratic hinge loss given by

.. math::
    \ell(y, y') =
    \begin{cases}
    \frac 12 (1 - y y')^2 &\text{ if } y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------


:class:`ModelSmoothedHinge <tick.optim.model.ModelSmoothedHinge>`
*****************************************************************

This is the smoothed hinge loss given by

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

:class:`ModelModifiedHuber <tick.optim.model.ModelModifiedHuber>`
*****************************************************************

The modified Huber loss, used for robust classification (less sensitive to
outliers). The loss is given by

.. math::
    \ell(y, y') =
    \begin{cases}
    - 4 y y' &\text{ if } y y' \leq -1 \\
    (1 - y y')^2 &\text{ if } -1 < y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

----------------------------------------

Count data models
-----------------

.. plot:: modules/code_samples/optim/plot_losses_count_data.py

:class:`ModelPoisReg <tick.optim.model.ModelPoisReg>`
*****************************************************

Poisson regression with exponential link with loss corresponds to the loss

.. math::
    \ell(y, y') = e^{y'} - y y'

for :math:`y \in \mathbb N` and :math:`y' \in \mathbb R` and is obtained
using ``link='exponential'``.

Poisson regression with identity link, namely with loss

.. math::
    \ell(y, y') = y' - y \log(y')

for :math:`y \in \mathbb N` and :math:`y' > 0` is obtained using
``link='identity'``.

----------------------------------------


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
Sorted L1, see :ref:`prox classes <prox>`) allows to detect outliers
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

===============================================================  ===============================
Model                                                            Class
===============================================================  ===============================
Least-squares for Hawkes model with exponential kernels          :class:`ModelHawkesFixedExpKernLeastSq <tick.optim.model.ModelHawkesFixedExpKernLeastSq>`
Log-likelihood for Hawkes model with exponential kernels         :class:`ModelHawkesFixedExpKernLogLik <tick.optim.model.ModelHawkesFixedExpKernLogLik>`
Least-squares for Hawkes model with sum of exponential kernels   :class:`ModelHawkesFixedSumExpKernLeastSq <tick.optim.model.ModelHawkesFixedSumExpKernLeastSq>`
Log-likelihood for Hawkes model with sum of exponential kernels  :class:`ModelHawkesFixedSumExpKernLogLik <tick.optim.model.ModelHawkesFixedSumExpKernLogLik>`
===============================================================  ===============================


4. What's under the hood?
=========================

All model classes have a ``loss`` and ``grad`` method, that are used by batch
algorithms to fit the model. These classes contains a C++ object, that does the
computations. Some methods are hidden within this C++ object, and are accessible
only through C++ (such as ``loss_i`` and ``grad_i`` that compute the gradient
using the single data point :math:`(x_i, y_i)`). These hidden methods are used
in the stochastic solvers, and are available through C++ only for efficiency.
