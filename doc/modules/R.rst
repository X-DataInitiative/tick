
.. _r_usage:

===================================================
How to use ``tick`` from the R statistical software
===================================================

``tick`` really is a Python library. However, you can use it quite easily
from the ``R`` statistical software using the ``reticulate`` library, that
builds automatically wrappers around Python code.

Note that the best experience and performance for using ``tick`` is
obviously from Python, the main problem being the fact that all two-dimensional
arrays will be duplicated, since ``R`` supports only the column-major format.
This copy is transparent for the end-user.

Before starting, you need to install the ``reticulate`` library in your ``R``
system, and have a working version of ``tick`` on your version of Python 3.
Your ``R`` code should begin with

.. code-block:: r

    library('reticulate')
    reticulate::use_python('SOMEPATH1/python3')
    tick = import_from_path('tick', 'SOMEPATH2/tick/')


where ``SOMEPATH1`` is the path to the Python 3 binary used on your system
and ``SOMEPATH2`` is the path to the installation of ``tick`` on your system
(path to the source code if you git cloned and compiled directly ``tick``).


.. _r_usage_logistic_regression:

1. Training a logistic regression model
=======================================

The example given below downloads a dataset for binary classification and
trains a logistic regression with elastic-net penalization.
Then it displays the ROC curve on testing data using some tools from
``sklearn.metric`` (that needs to be installed in your python distribution
as well).

.. code-block:: r

    # Fetch data
    fetch_tick_dataset = tick$dataset$fetch_tick_dataset
    train_set = fetch_tick_dataset('binary/adult/adult.trn.bz2')
    test_set = fetch_tick_dataset('binary/adult/adult.tst.bz2')

    # Logistic regression training
    LogisticRegression = tick$inference$LogisticRegression
    clf = LogisticRegression(penalty='elasticnet')
    clf$fit(train_set[[1]], train_set[[2]])

    # Plot of the ROC curve on the testing set using scikit-learn
    metrics = import('sklearn.metrics')
    predictions = clf$predict_proba(test_set[[1]])
    res = metrics$roc_curve(test_set[[2]], predictions[, 2])
    fpr = res[[1]]
    tpr = res[[2]]
    plot(fpr, tpr, type='l')

This is very close to the equivalent ``Python`` code. To produce this ``R``
code from the ``Python`` equivalent, we simply replaced ``.`` by ``$``
when accessing modules and classes, and when a function returns more than
one argument (such as ``metrics$roc_curve`` in the previous example), we access
the returned arguments through the ``res[[1]]`` syntax.


.. _r_usage_hawkes_simulation:

2. Simulation of a Hawkes process
=================================

This example simulates a unidimensional Hawkes process with sum of exponentials
kernel, and plots the simulated intensity.
Note that integer 0 must be coded as ``0L``, otherwise it is interpreted as
floating point and results in errors.

.. code-block:: r

    SimuHawkes = tick$simulation$SimuHawkes
    HawkesKernelSumExp = tick$simulation$HawkesKernelSumExp
    end_time = 40L

    hawkes = SimuHawkes(n_nodes=1L, end_time=end_time, verbose=FALSE, seed=1398L)
    kernel = HawkesKernelSumExp(c(.1, .2, .1), c(1., 3., 7.))
    hawkes$set_kernel(0L, 0L, kernel)
    hawkes$set_baseline(0L, 1.)

    dt = 0.01
    hawkes$track_intensity(dt)
    hawkes$simulate()
    timestamps = hawkes$timestamps
    intensity = hawkes$tracked_intensity
    intensity_times = hawkes$intensity_tracked_times

    plot(intensity[[1]], type='l', xlab='Time', ylab='Hawkes intensity')


.. _r_usage_tick_optim:

3. Using ``tick.optim`` from R
==============================

In the following example we use ``tick.simulation`` to simulate a linear
regression model with sparse weights and ``tick.optim`` to estimate these
weights, by combining a ``ModelLinReg`` object for linear regression, and a
``ProxSlope`` object for SLOPE penalization (Sorted-L1 norm) in a ``AGD``
solver, namely accelerated gradient descent.
We then plot the true weights and estimated ones, together with the solver's
history. Note that the estimation of the weights can be achieved more easily
through the ``tick.inference.LinearRegression`` class as well.

.. code-block:: r

    # Simulation of a linear regression model with sparse weights
    weights_sparse_gauss = tick$simulation$weights_sparse_gauss
    weights = weights_sparse_gauss(n_weights=50L)
    SimuLinReg = tick$simulation$SimuLinReg
    simu = SimuLinReg(weights=weights, n_samples=5000L)
    res = simu$simulate()
    X = res[[1]]
    y = res[[2]]

    # Use tick.optim to train a linear regression model with SLOPE penalization
    optim = tick$optim
    ModelLinReg = optim$model$ModelLinReg
    ProxSlope = optim$prox$ProxSlope
    AGD = optim$solver$AGD

    model = ModelLinReg(fit_intercept=FALSE)$fit(X, y)
    prox = ProxSlope(strength=1e-2, fdr=0.05)
    step = 1 / model$get_lip_best()
    solver = AGD(step=step)$set_model(model)$set_prox(prox)
    x_min = solver$solve()

    # Plot the true weights, estimated ones and solver's history
    par(mfrow=c(1, 3))
    plot(weights, type='h', ylab='Weights')
    title('Ground truth weights')
    plot(x_min, type='h', ylab='Weights')
    title('Learned weights')
    plot(solver$get_history('n_iter'), solver$get_history('obj'), type='l')
    title('Solver history')
