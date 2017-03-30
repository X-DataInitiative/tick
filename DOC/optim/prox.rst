
.. _proximal-operators:

Proximal operators [`tick.optim.prox`]
======================================

Introduction
------------

Module for the computation of proximal operators.
This module is heavily used throughout the library, and mostly
called from within solvers of the module `tick.optim.solver`.

The proximal operator of a convex
function :math:`g : \mathbb R^d \rightarrow \mathbb R^d` at some
point :math:`y\in \mathbb R^d` is defined as the unique minimizer of the
problem

.. math::
   \text{prox}_{g}(\theta, t) =\text{argmin}_{\theta' \in \mathbb R^d} \Big\{ \frac 12 \| \theta - \theta' \|_2^2 + t g(\theta') \Big\}

where :math:`\lambda > 0` is a regularization parameter. Note that in
the particular case where :math:`g(\theta) = \delta_{C}(\theta)`, with
:math:`C` a convex set, then :math:`\text{prox}_g` is a projection
operator (here :math:`\delta_{C}(\theta) = 0` if :math:`\theta \in C`
and :math:`+\infty` otherwise).

Available operators
-------------------

======================  =================================================================================================  ==============
Operator                Formula                                                                                            Class
======================  =================================================================================================  ==============
:ref:`prox-zero`        :math:`g(\theta) = 0`                                                                              `ProxZero`
:ref:`prox-positive`    :math:`g(\theta) = \delta_C(\theta)` where :math:`C=` set of vectors with non-negative entries     `ProxPositive`
:ref:`prox-l1`          :math:`g(\theta) = \sum_{j=1}^d |\theta_j|`                                                        `ProxL1`
:ref:`prox-l1w`         :math:`g(\theta) = \sum_{j=1}^d w_j |\theta_j|`                                                    `ProxL1w`
:ref:`prox-l2sq`        :math:`g(\theta) = \sum_{j=1}^d \frac{\theta_j^2}{2}`                                              `ProxL2Sq`
:ref:`prox-elasticnet`  :math:`g(\theta) = \sum_{j=1}^{d} \alpha |\theta_j| + (1 - \alpha) \frac{\theta_j^2}{2}`           `ProxElasticNet`
:ref:`prox-tv`          :math:`g(\theta) = \sum_{j=2}^d |\theta_j - \theta_{j-1}|`                                         `ProxTV`
:ref:`prox-nuclear`     :math:`g(\theta) = \sum_{j=1}^{q} \sigma_j(\theta)`                                                `ProxNuclear`
:ref:`prox-sorted_l1`   :math:`g(\theta) = \sum_{j=1}^{d} w_j |\theta_{(j)}|` where :math:`|\theta_{(j)}|` is decreasing   `ProxSortedL1`
======================  =================================================================================================  ==============


Example
-------
The next example shows a basic use of a proximal operator.::

   import numpy as np
   from bokeh.plotting import output_notebook, show
   from tick.optim.prox import ProxTV
   from tick.plot import stems

   output_notebook()

   n_coeffs = 100
   x = np.random.randn(n_coeffs)
   prox = ProxTV(0.5)
   stems([x, prox.call(x)])
