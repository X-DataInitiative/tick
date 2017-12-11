
.. _solver:

==================================
:mod:`tick.solver`: solver toolbox
==================================

This is a solver toolbox, which is used to train almost all models in ``tick``.
It features two types of solvers: deterministic and stochastic.
Deterministic solvers use a full pass over data at each iteration, while
stochastic solvers make ``epoch_size`` iterations within each iteration.
Most of the optimization problems considered here can be written as

.. math::
    \min_w f(w) + g(w)

where :math:`f` is a goodness-of-fit term, which depends on the model
considered, and :math:`g` is a function penalizing :math:`w` (see the
:ref:`tick.prox <prox>` module).

1. The ``solver`` class API
---------------------------

All the solvers have a ``set_model`` method, which corresponds to the function
:math:`f` to pass the model to be trained, and a ``set_prox`` method to pass
the penalization, which corresponds to the :math:`g`.
The solver is launched using the ``solve`` method to which a starting point and
eventually a step-size can be given. Here is an example, another example is
also given :ref:`below <solver-example>`.

.. testcode::

    import numpy as np
    from tick.simulation import SimuLogReg, weights_sparse_gauss
    from tick.solver import SVRG
    from tick.optim.model import ModelLogReg
    from tick.prox import ProxElasticNet

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


2. Available solvers
--------------------

Here is the list of the solvers available in ``tick``. Note that a lot of
details about each solver is available in the classes documentations, linked
below.

=======================================================  ========================================
Solver                                                   Class
=======================================================  ========================================
Proximal gradient descent                                :class:`GD <tick.solver.GD>`
Accelerated proximal gradient descent                    :class:`AGD <tick.solver.AGD>`
Broyden, Fletcher, Goldfarb, and Shannon (quasi-newton)  :class:`BFGS <tick.solver.BFGS>`
Self-Concordant Proximal Gradient Descent                :class:`SCPG <tick.solver.SCPG>`
Generalized Forward-Backward                             :class:`GFB <tick.solver.GFB>`
Stochastic Gradient Descent                              :class:`SGD <tick.solver.SGD>`
Adaptive Gradient Descent solver                         :class:`AdaGrad <tick.solver.AdaGrad>`
Stochastic Variance Reduced Descent                      :class:`SVRG <tick.solver.SVRG>`
Stochastic Averaged Gradient Descent                     :class:`SAGA <tick.solver.SAGA>`
Stochastic Dual Coordinate Ascent                        :class:`SDCA <tick.solver.SDCA>`
=======================================================  ========================================

.. _solver-example:

3. Example
----------

Here is an example of combination of a ``model`` a ``prox`` and a ``solver`` to
compare the training time of several solvers for logistic regression with the
elastic-net penalization.
Note that, we specify a ``range=(0, n_features)`` so that the intercept is not penalized
(see :ref:`tick.prox <prox>` for more details).

.. plot:: modules/code_samples/solver/plot_solver_comparison.py
    :include-source:
