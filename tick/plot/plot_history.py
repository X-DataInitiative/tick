# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from tick.base.learner import LearnerOptim
from tick.plot.plot_utilities import get_plot_color
from tick.solver.base import Solver


def extract_history(solvers, x, y, given_labels):
    x_arrays = []
    y_arrays = []
    labels = []
    for i, solver in enumerate(solvers):
        if isinstance(solver, LearnerOptim):
            history = solver._solver_obj.history
            label = solver._solver_obj.name
        elif isinstance(solver, Solver):
            history = solver.history
            label = solver.name
        else:
            raise ValueError('%s has no history' % solver.__class__.__name__)

        # if label was given we override it
        if given_labels is not None and i < len(given_labels):
            label = given_labels[i]

        if x not in history.values.keys():
            raise ValueError("%s has no history for %s" % (label, x))

        elif y not in history.values.keys():
            raise ValueError("%s has no history for %s" % (label, y))
        else:
            x_arrays.append(np.array(history.values[x]))
            y_arrays.append(np.array(history.values[y]))
            labels.append(label)

    return x_arrays, y_arrays, labels


def plot_history(solvers, x='n_iter', y='obj', labels=None, show=True,
                 log_scale: bool = False, dist_min: bool = False,
                 rendering: str = 'matplotlib', ax=None):
    """Plot the history of convergence of learners or solvers.
    
    It is used to compare easily their convergence performance.

    Parameters
    ----------
    solvers : `list` of `object` with and history to plot, namely solvers
        (children of `tick.solver.base.Solver`) or learners (children of
        `tick.hawkes.inference.base.LearnerOptim`)

    x : `str`, default='n_iter'
        - if 'n_iter' : iteration number
        - if 'time' : computation time

    y : `str`, default='obj'
        - if 'obj' : the objective (value of the function minimized).
            Other choices are possible, any of those present in the
            history

    labels : `list` of `str`, default=None
        Label of each solver in the legend. If set to None then the class
        name of each solver will be used.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    log_scale : `bool`, default=`False`
        If `True`, then y-axis is on a log-scale

    dist_min : `bool`, default=`False`
        If `True`, plot the difference between `y` of each solver and the
        minimal `y` of all solvers. This is useful when comparing solvers on
        a logarithmic scale, to illustrate linear convergence of algorithms

    rendering : {'matplotlib', 'bokeh'}, default='matplotlib'
        Rendering library. 'bokeh' might fail if the module is not installed.

    ax : `list` of `matplotlib.axes`, default=None
        If not None, the figure will be plot on this axis and show will be
        set to False. Used only with matplotlib
    """
    x_arrays, y_arrays, labels = extract_history(solvers, x, y, labels)

    if dist_min:
        min_y = np.min(np.hstack(y_arrays))
        y_arrays = [y_array - min_y for y_array in y_arrays]

    min_x, max_x = np.min(np.hstack(x_arrays)), np.max(np.hstack(x_arrays))
    min_y, max_y = np.min(np.hstack(y_arrays)), np.max(np.hstack(y_arrays))

    # We want to ensure theses plots starts at 0
    if x in ['time', 'n_iter']:
        min_x = 0

    if rendering == 'matplotlib':
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            show = False

        for i, (solver, x_array, y_array, label) in enumerate(
                zip(solvers, x_arrays, y_arrays, labels)):
            color = get_plot_color(i)
            ax.plot(x_array, y_array, lw=3, label=label, color=color)

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel(y, fontsize=16)
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

        if show is True:
            plt.show()

        return ax.figure

    elif rendering == 'bokeh':
        mins = (min_x, max_x, min_y, max_y)
        return plot_bokeh_history(solvers, x, y, x_arrays, y_arrays, mins,
                                  labels, log_scale, show)

    else:
        raise ValueError("Unknown rendering type. Expected 'matplotlib' or "
                         "'bokeh', received %s" % rendering)


def plot_bokeh_history(solvers, x, y, x_arrays, y_arrays, mins, legends,
                       log_scale, show):
    import bokeh.plotting as bk

    min_x, max_x, min_y, max_y = mins
    if log_scale:
        # Bokeh has a weird behaviour when using logscale with 0 entries...
        # We use the difference between smallest value of second small
        # to set the range of y
        all_ys = np.hstack(y_arrays)
        y_range_min = np.min(all_ys[all_ys != 0])
        if y_range_min < 0:
            raise ValueError("Cannot plot negative values on a log scale")

        fig = bk.Figure(plot_height=300, y_axis_type="log",
                        y_range=[y_range_min, max_y])
    else:
        fig = bk.Figure(plot_height=300, x_range=[min_x, max_x],
                        y_range=[min_y, max_y])

    for i, (solver, x_array, y_array, legend) in enumerate(
            zip(solvers, x_arrays, y_arrays, legends)):
        color = get_plot_color(i)
        fig.line(x_array, y_array, line_width=3, legend=legend, color=color)
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_size = "12pt"
    if show:
        bk.show(fig)
        return None
    else:
        return fig
