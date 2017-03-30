Time functions
==============

This example illustrates the use of `tick.base.TimeFunction`.

.. contents::
    :depth: 3
    :backlinks: none


.. testsetup:: *

    from tick import base
    import numpy as np
    from tick.base import TimeFunction
    from pylab import rcParams
    from tick import simulation
    from scipy import stats
    import matplotlib.pyplot as plt

    rcParams['figure.figsize'] = 20, 4

Basics
------
Here is an example using a time function which takes values
of array Y at times of array T

.. testcode:: [linear_timefunction]

    T = np.array([0, 1, 2, 5], dtype=float)
    Y = np.array([2, 4.1, 1, 2], dtype=float)

    linear_timefunction = TimeFunction([T, Y])


By default the time function will give a linear interpolation
from the two nearest points for any time value, and it equals
0 outside of its bounds

.. doctest:: [linear_timefunction]

    >>> linear_timefunction.value(2)
    1.00000...
    >>> linear_timefunction.value(3)
    1.33333...


.. doctest:: [linear_timefunction]

    >>> linear_timefunction.value(-1)
    0.0
    >>> linear_timefunction.value(7)
    0.0

Plotting
--------
Now we plot several time functions

1. linear interpolation and big dt, where dt is the step
   between two points kept for our interpolation
2. constant on right interpolation and continuous border
3. constant on left interpolation and border fixed to a given value


This code will produce graphs of the above described functions

.. plot:: z_tutorials/misc/code_samples/time_function_example.py
    :include-source:


.. warning::

    If the value for dt is not well chosen, then you could receive
    incorrect values for any interpolation in between steps.

    This behavior is illustrated in the first graph under the second point