Stochastic solvers
==================

This example illustrates the use of several solvers, including
`tick.optim.solver.SGD`, `tick.optim.solver.SVRG`, and
`tick.optim.solver.AGD`.

.. contents::
    :depth: 2
    :backlinks: none


.. testsetup:: *

    import numpy as np
    from tick.optim.solver import SVRG
    from tick.simulation import SimuLogReg
    from tick.optim.model import ModelLogReg
    from tick.optim.prox import ProxL2Sq
    np.set_printoptions(precision=4)


Solve
-----

Simulation
~~~~~~~~~~

First of all we generate data for logistic regression with given feature vector
`w` and intercept `c`.

.. testcode:: [stochastic_solver]

    n_features, n_samples = 5, 10000
    np.random.seed(732)
    w = np.random.normal(0, 1, n_features)
    c = 0.2

    sim = SimuLogReg(weights=w, intercept=c, n_samples=n_samples, seed=732, verbose=False)
    sim.simulate()
    X = sim.features
    y = sim.labels


Problem initialization
~~~~~~~~~~~~~~~~~~~~~~

We then create the associated model and the prox that we will use in our
solvers.

.. testcode:: [stochastic_solver]

    model = ModelLogReg(fit_intercept=True).fit(X, labels=y)
    l_l2sq = 1e-7
    prox = ProxL2Sq(strength=l_l2sq)


Solution
~~~~~~~~

.. testcode:: [stochastic_solver]

    svrg = SVRG(max_iter=1000, verbose=False, tol=1e-10)
    svrg.set_model(model)
    svrg.set_prox(prox)
    svrg.solve(np.zeros(n_features + 1), 0.1)


.. doctest:: [stochastic_solver]

    >>> print(w, c)
    [-1.9505  1.8576  0.2318  0.2274  0.167 ] 0.2
    >>> print(svrg.solution)
    [-1.9727  1.8532  0.2636  0.2135  0.1936  0.1806]


Plot
----
We can also easily plot and evaluate different solvers on the same problem

.. plot:: z_tutorials/optimization/code_samples/solvers_convergence.py
    :include-source:
