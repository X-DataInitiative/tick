

.. _robust:

==================
:mod:`tick.robust`
==================

This module provides tools for robust inference of generalized linear models
and outliers detection. It proposes two set of things: a learner for least-squares
regression with individual intercepts, see :ref:`robust-intercepts` and a set of
losses for robust supervised learning, see :ref:`robust-losses`.

.. _robust-intercepts:

1. Regression with individual intercepts
========================================

In this section, we describe the approach used in the
:class:`RobustLinearRegression <tick.robust.RobustLinearRegression>` learner.
It allows to detect outliers and to fit a least-squares regression
model at the same time. Namely, given training data
:math:`(x_i, y_i) \in \mathbb R^d \times \mathbb R` for :math:`i=1, \ldots, n`,
it considers the following problem

.. math::
    \frac 1n \sum_{i=1}^n \ell(y_i, x_i^\top w + b + \mu_i) + s \sum_{j=1}^{d}
    c_j | w_{(j)} | + g(w),

where :math:`|w_{(j)}|` is decreasing, :math:`w \in \mathbb R^d` is a vector
containing the model weights, :math:`\mu = [\mu_1 \cdots \mu_n] \in \mathbb R^n`
is a vector containing individual intercepts, :math:`b \in \mathbb R` is the
population intercept, :math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is the
least-squares loss and :math:`g` is a penalization function for the model weights, for
which different choices are possible, see :ref:`prox`.
Note that in this problem the vector of individual intercepts :math:`\mu` is
penalized by a sorted-L1 norm, also called SLOPE, see :ref:`prox` for details,
where the weights are given by

.. math::
    w_j = \Phi \Big( 1 - \frac{j \alpha}{2 n} \Big),

where :math:`\alpha` stands for the FDR level for the support detection of
:math:`\mu`, that can be tuned with the ``fdr`` parameter.
The global penalization level :math:`s` corresponds to
the inverse of the ``C_sample_intercepts`` parameter.

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RobustLinearRegression

Tools for robust estimation of the standard-deviation
-----------------------------------------------------
Some tools for the robust estimation of the standard deviation are also provided.

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: function.rst

   std_mad
   std_iqr

Example
-------
.. plot:: ../examples/plot_robust_linear_regression.py
    :include-source:

.. _robust-losses:

2. Robust losses
================

Tick also provides losses for robust inference when it is believed that data contains
outliers. It also provides the model
:class:`ModelLinRegWithIntercepts <tick.robust.ModelLinRegWithIntercepts>` which is
used in the :class:`RobustLinearRegression <tick.robust.RobustLinearRegression>`
described above.

========================================  ==============  ==========  ==========================================
Model                                     Type            Label type  Class
========================================  ==============  ==========  ==========================================
Linear regression with intercepts         Regression      Continuous  :class:`ModelLinRegWithIntercepts <tick.robust.ModelLinRegWithIntercepts>`
Huber regression                          Regression      Continuous  :class:`ModelHuber <tick.robust.ModelHuber>`
Epsilon-insensitive regression            Regression      Continuous  :class:`ModelEpsilonInsensitive <tick.robust.ModelEpsilonInsensitive>`
Absolute regression                       Regression      Continuous  :class:`ModelAbsoluteRegression <tick.robust.ModelAbsoluteRegression>`
Modified Huber loss                       Classification  Binary      :class:`ModelModifiedHuber <tick.robust.ModelModifiedHuber>`
========================================  ==============  ==========  ==========================================

The robust losses are illustrated in the following picture, among the other losses
provided in the :ref:`linear_model` module.

.. plot:: modules/code_samples/plot_linear_model_losses.py


:class:`ModelHuber <tick.robust.ModelHuber>`
--------------------------------------------

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

:class:`ModelEpsilonInsensitive <tick.robust.ModelEpsilonInsensitive>`
----------------------------------------------------------------------

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

:class:`ModelAbsoluteRegression <tick.robust.ModelAbsoluteRegression>`
----------------------------------------------------------------------

The L1 loss given by

.. math::
    \ell(y, y') = |y' - y|

for :math:`y, y' \in \mathbb R`

----------------------------------------

:class:`ModelModifiedHuber <tick.robust.ModelModifiedHuber>`
------------------------------------------------------------

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

