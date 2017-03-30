Optimization toolbox
====================

.. toctree::
   :maxdepth: 2

   model
   prox
   solver


We are in the following case

.. math::
    \min_x f(x) + g(x)

In which :math:`f` and :math:`g` are usually convex and :math:`g` is
prox-capable. Depending on the problem, you might want to use a specific
algorithm to solve it.

In tick:

* we call :math:`f` the **model**. The model gives information about a
  statistical model based on the data our problem relies on.
  Depending on the case, it gives first order information (loss, gradient) or
  second order information (hessian related values).

  * linear regression `tick.optim.model.ModelLinReg`
  * logistic regression `tick.optim.model.ModelLogReg`

* we call :math:`g` the **prox**. The prox will make our solution looks like
  what we think it should.

  * Prox L1 (or lasso) `tick.optim.prox.ProxL1`
  * Prox L2 (or ridge) `tick.optim.prox.ProxL2Sq`

* we call the algorithm the **solver**. The solver associated the model and
  the prox use an algorithm to determine what is the minimizer of our problem.
  Examples of solver are SGD (stochastic gradient descent), AGD or SVRG.

  * SGD `tick.optim.solver.SGD`
  * AGD `tick.optim.solver.AGD`
  * SVRG `tick.optim.solver.SVRG`


This allow us to try many different combinations of models, proximal operators
and solvers.

Here is a quick example of finding the result of a logistic regression with a
ridge penalization thanks to SVRG solver:


.. testcode::

    import numpy as np
    from tick.optim.model import ModelLogReg
    from tick.optim.solver import SVRG
    from tick.optim.prox import ProxL2Sq
    from tick.simulation import SimuLogReg

    # simulate logistic data
    w = np.array([-2, 0, 4, 2, 1])
    X, y = SimuLogReg(w, n_samples=10000, verbose=False).simulate()

    # create logistic model
    model = ModelLogReg(fit_intercept=False)

    # make model fits the data
    model.fit(X, y)

    # Create the prox L2
    l_l2sq = 1e-7
    prox = ProxL2Sq(l_l2sq)

    # Create the solver
    svrg = SVRG(print_every=3, max_iter=9)

    # Make solver work with our model and prox
    svrg.set_model(model)
    svrg.set_prox(prox)

    # Find the solution
    minimizer = svrg.solve(np.zeros(model.n_coeffs), 0.1)
    print("\nfound minimizer\n", minimizer)

This outputs:

.. testoutput::
    :hide:
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Launching the solver SVRG...
      n_iter  |    obj    |  rel_obj
            0 |    ...    |    ...
            3 |    ...    |    ...
            6 |    ...    |    ...
            9 |    ...    |    ...
    Done solving using SVRG in ... seconds

    found minimizer
     [...  ...  ...  ...  ...]

.. code-block:: none

    Launching the solver SVRG...
      n_iter  |    obj    |  rel_obj
            0 |  2.57e-01 |  6.29e-01
            3 |  2.48e-01 |  5.87e-05
            6 |  2.48e-01 |  2.37e-08
            9 |  2.48e-01 |  1.67e-12
    Done solving using SVRG in 0.06659603118896484 seconds

    found minimizer
     [-1.93296625  0.01182093  3.89796454  1.95714016  0.9914066 ]