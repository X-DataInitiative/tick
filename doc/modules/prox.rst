
.. _prox:

====================================
:mod:`tick.prox`: proximal operators
====================================

This module proposes a large number of proximal operator, allowing the use
many penalization techniques for model fitting. Namely, most optimization
problems considered in ``tick`` (but not all) can be written as

.. math::
    \min_w f(w) + g(w)

where :math:`f` is a goodness-of-fit term and :math:`g` is a function
penalizing :math:`w`. Depending on the problem, you might want to use some
function :math:`g` in order to induce a specific property of the weights
:math:`w` of the model.

The proximal operator of a convex function :math:`g` at some point :math:`w`
is defined as the unique minimizer of the problem

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
an intercept, which is not penalized, see :ref:`generalized linear models <linear-models>`.
Indeed, in all ``prox`` classes, an optional ``range`` parameter is available, to apply
the regularization only to a subset of entries of :math:`w`.

1. The ``prox`` class API
-------------------------

Let us describe the ``prox`` API with the :class:`ProxL1<tick.prox.ProxL1>`
class, that provides the proximal operator of the function :math:`g(w) = s \|w\|_1 = s \sum_{j=1}^d |w_j|`.


.. testcode:: [optim-model-prox]

    import numpy as np
    from tick.prox import ProxL1

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

.. plot:: modules/code_samples/prox/plot_prox_api.py
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


2. Available operators
----------------------

The list of available operators in ``tick`` given in the next table.

=======================  ===========================================================================================================  ==============
Penalization             Function                                                                                                     Class
=======================  ===========================================================================================================  ==============
Identity                 :math:`g(w) = 0`                                                                                             :class:`ProxZero <tick.prox.ProxZero>`
L1 norm                  :math:`g(w) = s \sum_{j=1}^d |w_j|`                                                                          :class:`ProxL1 <tick.prox.ProxL1>`
L1 norm with weights     :math:`g(w) = s \sum_{j=1}^d c_j |w_j|`                                                                      :class:`ProxL1w <tick.prox.ProxL1w>`
Ridge                    :math:`g(w) = s \sum_{j=1}^d \frac{w_j^2}{2}`                                                                :class:`ProxL2Sq <tick.prox.ProxL2Sq>`
L2 norm                  :math:`g(w) = s \sqrt{d \sum_{j=1}^d w_j^2}`                                                                 :class:`ProxL2 <tick.prox.ProxL2>`
Elastic-net              :math:`g(w) = s \Big(\sum_{j=1}^{d} \alpha |w_j| + (1 - \alpha) \frac{w_j^2}{2} \Big)`                       :class:`ProxElasticNet <tick.prox.ProxElasticNet>`
Nuclear norm             :math:`g(w) = s \sum_{j=1}^{q} \sigma_j(w)`                                                                  :class:`ProxNuclear <tick.prox.ProxNuclear>`
Non-negative constraint  :math:`g(w) = s \delta_C(w)` where :math:`C=` set of vectors with non-negative entries                       :class:`ProxPositive <tick.prox.ProxPositive>`
Equality constraint      :math:`g(w) = s \delta_C(w)` where :math:`C=` set of vectors with identical entries                          :class:`ProxEquality <tick.prox.Equality>`
Sorted L1                :math:`g(w) = s \sum_{j=1}^{d} c_j |w_{(j)}|` where :math:`|w_{(j)}|` is decreasing                          :class:`ProxSlope <tick.prox.ProxSlope>`
Total-variation          :math:`g(w) = s \sum_{j=2}^d |w_j - w_{j-1}|`                                                                :class:`ProxTV <tick.prox.ProxTV>`
Binarsity                :math:`g(w) = s \sum_{j=1}^d \big( \sum_{k=2}^{d_j} |w_{j,k} - w_{j,k-1} | + \delta_C(w_{j,\bullet}) \big)`  :class:`ProxBinarsity <tick.prox.ProxBinarsity>`
Group L1                 :math:`g(w) = s \sum_{j=1}^d \sqrt{d_j} \| w^{(j)}\|_2`                                                      :class:`ProxGroupL1 <tick.prox.ProxGroupL1>`
=======================  ===========================================================================================================  ==============

Another ``prox`` class is the :class:`ProxMulti <tick.prox.ProxMulti>` that allows
to combine any proximal operators together.
It simply applies sequentially each operator passed to :class:`ProxMulti <tick.prox.ProxMulti>`,
one after the other. Here is an example of combination of a total-variation penalization and L1 penalization
applied to different parts of a vector.

.. plot:: modules/code_samples/prox/plot_prox_multi.py
    :include-source:

3. Example
----------
Here is an illustration of the effect of these proximal operators on an example.

.. plot:: ../examples/plot_prox_example.py
    :include-source:
