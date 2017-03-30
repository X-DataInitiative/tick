Poisson processes simulation
============================

This example illustrates the use of `tick.simulation.SimuInhomogeneousPoisson`
and `tick.simulation.SimuPoissonProcess` which are classes that allow us to simulate
these two kinds of processes.

.. contents::
    :depth: 3
    :backlinks: none


Poisson process with constant intensity
---------------------------------------

.. plot:: z_tutorials/simulation/code_samples/poisson_constant_intensity.py
    :include-source:


Inhomogeneous Poisson processes
-------------------------------

Now we plot a Poisson process with varying intensity

.. plot:: z_tutorials/simulation/code_samples/poisson_simple_homogeneous.py
    :include-source:

We see that the higher the intensity is, the more jumps we have.
