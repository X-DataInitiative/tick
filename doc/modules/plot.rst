

.. _plot:

====================================
:mod:`tick.plot`: plotting utilities
====================================

*tick* provides some plot routines able to achieve common plots from tick
objects.

.. contents::
    :depth: 3
    :backlinks: none

Optimization
------------

Used to compare efficiency of optimization algorithms implemented in
`tick.solver`.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot.plot_history

.. plot:: modules/code_samples/solver/plot_solver_comparison.py
    :include-source:

Hawkes estimation
-----------------

Used to observe hawkes parameters obtained by hawkes learners.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_hawkes_kernels
   plot.plot_hawkes_kernel_norms
   plot.plot_basis_kernels

.. plot:: modules/code_samples/plot/plot_hawkes_matrix_exp_kernels.py
    :include-source:

Point process simulation
------------------------

Used to observe the behavior of a point process simulation.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_point_process

.. plot:: ../examples/plot_poisson_inhomogeneous.py
    :include-source:

Others
------

Some other plot utilities.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_timefunction
   plot.stems

.. plot:: modules/code_samples/simulation/plot_time_function.py
    :include-source:
