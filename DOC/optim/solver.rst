
.. _solvers:

Solvers [`tick.optim.solver`]
=============================

Introduction
------------

Module that contains all solvers available in `tick`. This module is
used throughout the library. "Batch" or "full" solvers
use a full pass over data at each iteration, while "stochastic" solvers
make ``epoch_size`` iterations using a single data-point at each
iteration.


Batch solvers
-------------

======================    ==============
Solver                    Class
======================    ==============
:ref:`gd`                 `GD`
:ref:`agd`                `AGD`
:ref:`bfgs`               `BFGS`
:ref:`scpg`               `SCPG`
======================    ==============

Stochastic solvers
------------------

======================    ==============
Solver                    Class
======================    ==============
:ref:`sgd`                `SGD`
:ref:`svrg`               `SVRG`
:ref:`sdca`               `SDCA`
:ref:`adagrad`            `AdaGrad`
======================    ==============