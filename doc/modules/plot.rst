

.. _plot:

================
:mod:`tick.plot`
================

This module gathers all the plotting utilities of `tick`, namely plots for a
solver's history, see :ref:`plot-optim`, plots for Hawkes processes, see :ref:`plot-hawkes`,
plots for point process simulations :ref:`plot-pp` and some other useful plots, see
:ref:`plot-misc`.

.. _plot-optim:

1. History plot
---------------

This plot is used to compare the efficiency of optimization algorithms
implemented in :mod:`tick.solver`.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot.plot_history

Example
*******

.. plot:: modules/code_samples/plot_solver_comparison.py
    :include-source:

.. _plot-hawkes:

2. Plots for Hawkes processes
-----------------------------

These plots are used to observe hawkes parameters obtained by hawkes learners.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_hawkes_kernels
   plot.plot_hawkes_kernel_norms
   plot.plot_basis_kernels

Example
*******

.. plot:: modules/code_samples/plot_hawkes_matrix_exp_kernels.py
    :include-source:

.. _plot-pp:

3. Plots for point process simulation
-------------------------------------

This plot is used to plot a point process simulation.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_point_process

Example
*******

.. plot:: ../examples/plot_poisson_inhomogeneous.py
    :include-source:

.. _plot-misc:

4. Miscellaneous plots
----------------------

Some other plots are particularly useful: plots for :class:`TimeFunction <tick.base.TimeFunction>``
and a plot generating several stem plots at the same time, allowing `matplotlib`
or `bokeh` rendering.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_timefunction
   plot.stems

Example
*******

.. plot:: modules/code_samples/plot_time_function.py
    :include-source:
