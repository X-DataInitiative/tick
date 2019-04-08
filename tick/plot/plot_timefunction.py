# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from tick.base import TimeFunction


def _extended_discrete_xaxis(x_axis, n_points=100, eps=0.10):
    """Takes `x_axis` and returns an uniformly sampled array of values from
    its minimum to its maximum with few extra points

    Parameters
    ----------
    x_axis : `np.ndarray`
        Original x axis

    n_points : `int`, default=100
        The number of points in the sample

    eps : `float`, default=0.05
        Number of extra points in percentage

    Returns
    -------
    sample : `np.ndarray`
        The extended axis sample 
    """
    min_value = np.min(x_axis)
    max_value = np.max(x_axis)
    distance = max_value - min_value
    return np.linspace(min_value - eps * distance, max_value + eps * distance,
                       num=n_points)


def plot_timefunction(time_function, labels=None, n_points=300, show=True,
                      ax=None):
    """Quick plot of a `tick.base.TimeFunction`
    
    Parameters
    ----------
    time_function : `TimeFunction`
        The `TimeFunction` that will be plotted
    
    labels : `list` of `str`, default=None
        Labels that will be given to the plot. If None, labels will be
        automatically generated.

    n_points : `int`, default=300
        Number of points that will be used in abscissa. More points will lead
        to a more precise graph.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    ax : `matplotlib.axes`, default=None
        If not None, the figure will be plot on this axis and show will be
        set to False.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        show = False

    if time_function.is_constant:
        if labels is None:
            labels = ['value = %.3g' % time_function.border_value]

        t_values = np.arange(10).astype('float')
        ax.plot(t_values, time_function.value(t_values), label=labels[0])

    else:
        if labels is None:
            interpolation_to_legend = {
                TimeFunction.InterLinear: 'Linear',
                TimeFunction.InterConstLeft: 'Constant on left',
                TimeFunction.InterConstRight: 'Constant on right'
            }

            border_to_legend = {
                TimeFunction.Border0:
                    'border zero',
                TimeFunction.BorderConstant:
                    'border constant at %.3g' % time_function.border_value,
                TimeFunction.BorderContinue:
                    'border continue',
                TimeFunction.Cyclic:
                    'cyclic'
            }

            labels = [
                'original points',
                '%s and %s' %
                (interpolation_to_legend[time_function.inter_mode],
                 border_to_legend[time_function.border_type])
            ]

        original_t = time_function.original_t
        if time_function.border_type == TimeFunction.Cyclic:
            cycle_length = original_t[-1]
            original_t = np.hstack((original_t, original_t + cycle_length,
                                    original_t + 2 * cycle_length))

        t_values = _extended_discrete_xaxis(original_t, n_points=n_points)

        ax.plot(time_function.original_t, time_function.original_y, ls='',
                marker='o', label=labels[0])
        ax.plot(t_values, time_function.value(t_values), label=labels[1])

    ax.legend()
    if show is True:
        plt.show()

    return ax.figure
