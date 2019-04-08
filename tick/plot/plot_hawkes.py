# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tick.plot.plot_utilities import share_x, share_y


def plot_hawkes_kernel_norms(kernel_object, show=True, pcolor_kwargs=None,
                             node_names=None, rotate_x_labels=0.):
    """Generic function to plot Hawkes kernel norms.

    Parameters
    ----------
    kernel_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_kernel_norms()` : must return a 2d numpy
          array with the norm of each kernel

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    pcolor_kwargs : `dict`, default=`None`
        Extra pcolor kwargs such as cmap, vmin, vmax

    node_names : `list` of `str`, shape=(n_nodes, ), default=`None`
        node names that will be displayed on axis.
        If `None`, node index will be used.

    rotate_x_labels : `float`, default=`0.`
        Number of degrees to rotate the x-labels clockwise, to prevent 
        overlapping.

    Notes
    -----
    Kernels are displayed such that it shows norm of column influence's
    on row.
    """
    n_nodes = kernel_object.n_nodes

    if node_names is None:
        node_names = range(n_nodes)
    elif len(node_names) != n_nodes:
        ValueError('node_names must be a list of length {} but has length {}'
                   .format(n_nodes, len(node_names)))

    row_labels = ['${} \\rightarrow$'.format(i) for i in node_names]
    column_labels = ['$\\rightarrow {}$'.format(i) for i in node_names]

    norms = kernel_object.get_kernel_norms()
    fig, ax = plt.subplots()

    if rotate_x_labels != 0.:
        # we want clockwise rotation because x-axis is on top
        rotate_x_labels = -rotate_x_labels
        x_label_alignment = 'right'
    else:
        x_label_alignment = 'center'

    if pcolor_kwargs is None:
        pcolor_kwargs = {}

    if norms.min() >= 0:
        pcolor_kwargs.setdefault("cmap", plt.cm.Blues)
    else:
        # In this case we want a diverging colormap centered on 0
        pcolor_kwargs.setdefault("cmap", plt.cm.RdBu)
        max_abs_norm = np.max(np.abs(norms))
        pcolor_kwargs.setdefault("vmin", -max_abs_norm)
        pcolor_kwargs.setdefault("vmax", max_abs_norm)

    heatmap = ax.pcolor(norms, **pcolor_kwargs)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(norms.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(norms.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, fontsize=17, 
                       rotation=rotate_x_labels, ha=x_label_alignment)
    ax.set_yticklabels(column_labels, minor=False, fontsize=17)

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    fig.colorbar(heatmap, cax=cax)

    if show:
        plt.show()

    return fig


def plot_hawkes_kernels(kernel_object, support=None, hawkes=None, n_points=300,
                        show=True, log_scale=False, min_support=1e-4, ax=None):
    """Generic function to plot Hawkes kernels.

    Parameters
    ----------
    kernel_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_kernel_supports()` : must return a 2d numpy
          array with the size of the support of each kernel
        * `kernel_object.get_kernel_values(self, i, j, abscissa_array)` :
          must return as a numpy 1d array the sampled `(i,j)` kernel values
          corresponding to the abscissa `abscissa_array`

    support : `float`, default=None
        the size of the support that will be used to plot all the kernels.
        If None or non positive then the maximum kernel supports is used

    hawkes : `SimuHawkes`, default=None
        If a `SimuHawkes` object is given then the kernels plots are superposed
        with those of this object (considered as the `True` kernels). This is
        used to plot on the same plots the estimated kernels along with
        the true kernels.

    n_points : `int`, default=300
        Number of points that will be used in abscissa. More points will lead
        to a more precise graph.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    log_scale : `bool`, default=`False`
        If `True`, then x-axis and y-axis are on a log-scale. This is useful
        to plot power-law kernels.

    min_support : `float`, default=1e-4
        Start value of the plot. Only used if log_scale is `True`.
        
    ax : `np.ndarray` of `matplotlib.axes`, default=None
        If not None, the figure will be plot on these axes and show will be
        set to False.
    """
    if support is None or support <= 0:
        plot_supports = kernel_object.get_kernel_supports()
        support = plot_supports.max() * 1.2

    n_nodes = kernel_object.n_nodes

    if log_scale:
        x_values = np.logspace(
            np.log10(min_support), np.log10(support), n_points)
    else:
        x_values = np.linspace(0, support, n_points)

    if ax is None:
        fig, ax_list_list = plt.subplots(n_nodes, n_nodes, sharex=True,
                                         sharey=True)
    else:
        if ax.shape != (n_nodes, n_nodes):
            raise ValueError('Given ax has shape {} but should have shape {}'
                             .format(ax.shape, (n_nodes, n_nodes)))
        ax_list_list = ax
        show = False

    if n_nodes == 1:
        ax_list_list = np.array([[ax_list_list]])

    for i, ax_list in enumerate(ax_list_list):
        for j, ax in enumerate(ax_list):
            y_values = kernel_object.get_kernel_values(i, j, x_values)
            ax.plot(x_values, y_values, label="Kernel (%d, %d)" % (i, j))

            if hawkes:
                y_true_values = hawkes.kernels[i, j].get_values(x_values)
                ax.plot(x_values, y_true_values,
                        label="True Kernel (%d, %d)" % (i, j))

            # set x_label for last line
            if i == n_nodes - 1:
                ax.set_xlabel(r"$t$", fontsize=18)

            ax.set_ylabel(r"$\phi^{%g,%g}(t)$" % (i, j), fontsize=18)

            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')

            legend = ax.legend()
            for label in legend.get_texts():
                label.set_fontsize(12)

    if show:
        plt.show()

    return ax_list_list.ravel()[0].figure


def plot_hawkes_baseline_and_kernels(
        hawkes_object, kernel_support=None, hawkes=None, n_points=300,
        show=True, log_scale=False, min_support=1e-4, ax=None):
    """Generic function to plot Hawkes baseline and kernels.

    Parameters
    ----------
    hawkes_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_baseline_values(self, i, abscissa_array)` :
          must return as a numpy 1d array the sampled `i` baseline values
          corresponding to the abscissa `abscissa_array`
        * `kernel_object.period_length` : a field that stores the size of the 
          baseline period
        * `kernel_object.get_kernel_supports()` : must return a 2d numpy
          array with the size of the support of each kernel
        * `kernel_object.get_kernel_values(self, i, j, abscissa_array)` :
          must return as a numpy 1d array the sampled `(i,j)` kernel values
          corresponding to the abscissa `abscissa_array`

    kernel_support : `float`, default=None
        the size of the support that will be used to plot all the kernels.
        If None or non positive then the maximum kernel supports is used

    hawkes : `SimuHawkes`, default=None
        If a `SimuHawkes` object is given then the baseline and kernels plots 
        are superposed with those of this object (considered as the `True` 
        baseline and kernels). This is used to plot on the same plots the 
        estimated value along with the true values.

    n_points : `int`, default=300
        Number of points that will be used in abscissa. More points will lead
        to a more precise graph.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    log_scale : `bool`, default=`False`
        If `True`, then x-axis and y-axis of kernels are on a log-scale. 
        This is useful to plot power-law kernels.

    min_support : `float`, default=1e-4
        Start value of the kernels plot. Only used if log_scale is `True`.

    ax : `np.ndarray` of `matplotlib.axes`, default=None
        If not None, the figure will be plot on these axes and show will be
        set to False.
    """
    n_nodes = hawkes_object.n_nodes

    if ax is None:
        fig, ax_list_list = plt.subplots(n_nodes, n_nodes + 1, figsize=(10, 6))
    else:
        ax_list_list = ax
        show = False

    # invoke plot_hawkes_kernels
    ax_kernels = ax_list_list[:, 1:]
    plot_hawkes_kernels(hawkes_object, support=kernel_support, hawkes=hawkes,
                        n_points=n_points, show=False, log_scale=log_scale,
                        min_support=min_support, ax=ax_kernels)
    share_x(ax_kernels)
    share_y(ax_kernels)

    # plot hawkes baselines
    ax_baselines = ax_list_list[:, 0]
    t_values = np.linspace(0, hawkes_object.period_length, n_points)
    for i in range(n_nodes):
        ax = ax_baselines[i]
        ax.plot(t_values, hawkes_object.get_baseline_values(i, t_values),
                label='baseline ({})'.format(i))
        ax.plot(t_values, hawkes.get_baseline_values(i, t_values),
                label='true baseline ({})'.format(i))
        ax.set_ylabel("$\mu_{}(t)$".format(i), fontsize=18)

        # set x_label for last line
        if i == n_nodes - 1:
            ax.set_xlabel(r"$t$", fontsize=18)

        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(12)

    share_x(ax_baselines.reshape(2, 1))
    share_y(ax_baselines.reshape(2, 1))

    if show:
        plt.show()

    return ax_list_list.ravel()[0].figure


def _normalize_functions(y_values_list, t_values):
    """Normalize list of functions by their integral value

    Parameters
    ----------
    y_values_list : `list` of np.ndarray
        y values of the list of function we want to normalize

    t_values : `np.ndarray`
        t values shared by all functions given with y_values_list

    Returns
    -------
    normalized_y_values_list : `list` of np.ndarray
        Normalized y values of the given list of function

    normalizations : `np.ndarray`
        Normalization factors that have been used
    """
    y_values_list = np.array(y_values_list)
    normalizations = [
        1. / np.trapz(y_values, t_values) for y_values in y_values_list
    ]
    normalized_y_values_list = (y_values_list.T * normalizations).T
    return normalized_y_values_list, normalizations


def _find_best_match(diff_matrix):
    """Find best best possible match by induction

    Parameters
    ----------
    diff_matrix : `np.ndarray`, shape=(n_basis, n_basis)
        Matrix containing differences for all pairs of values

    Returns
    -------
    matches : `list`, shape=(n_nodes,)
        List of all found matches
    """
    diff_matrix = diff_matrix.astype(float)
    matches = []
    n_basis = diff_matrix.shape[0]
    for _ in range(n_basis):
        row, col = np.unravel_index(np.argmin(diff_matrix), (n_basis, n_basis))
        diff_matrix[row, :] = np.inf
        diff_matrix[:, col] = np.inf
        matches += [(row, col)]
    return matches


def plot_basis_kernels(learner, support=None, basis_kernels=None, n_points=300,
                       show=True):
    """Function used to plot basis of kernels
    
    It is used jointly with `tick.hawkes.inference.HawkesBasisKernels` learner class.

    Parameters
    ----------
    learner : `HawkesBasisKernels`
        The given learner which basis kernels are plotted

    support : `float`, default=None
        the size of the support that will be used to plot all the kernels.
        If None or non positive then the maximum kernel supports is used

    basis_kernels : `list` of `func`, default=None
        True basis kernels. If not `None`, it will find the closest estimated
        basis kernel and will plot it together.
        This basis kernels will be normalized to fit better with their
        estimations.

    n_points : `int`, default=300
        Number of points that will be used in abscissa. More points will lead
        to a more precise graph.

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.
    """
    if support is None or support <= 0:
        support = learner.kernel_support

    fig, ax_list = plt.subplots(1, learner.n_basis, figsize=(8, 4),
                                sharey=True)

    if basis_kernels is not None:
        if len(basis_kernels) != learner.n_basis:
            raise ValueError('Learner has {} basis kernels, cannot '
                             'compare to {} basis kernels'.format(
                                 learner.n_basis, len(basis_kernels)))

        t_values = learner.kernel_discretization[:-1]

        basis_values_list = [
            basis_kernel(t_values) for basis_kernel in basis_kernels
        ]
        normalized_basis_kernels, basis_normalizations = \
            _normalize_functions(basis_values_list, t_values)

        normalized_estimates, estimated_normalizations = \
            _normalize_functions(learner.basis_kernels, t_values)

        kernel_diff = np.array([[
            np.trapz(np.abs(nbf - ne), t_values)
            for nbf in normalized_basis_kernels
        ] for ne in normalized_estimates])

        matches = _find_best_match(kernel_diff)

        t_values = np.linspace(0, support, n_points)
        for estimated_index, basis_index in matches:
            basis_kernel = basis_kernels[basis_index]
            estimated = learner.basis_kernels[estimated_index]

            piecewise_y = np.repeat(estimated, 2)
            piecewise_t = np.hstack(
                (learner.kernel_discretization[0],
                 np.repeat(learner.kernel_discretization[1:-1], 2),
                 learner.kernel_discretization[-1]))

            ax_list[basis_index].step(piecewise_t, piecewise_y,
                                      label="estimated %i" % estimated_index)
            rescaled_basis = basis_kernel(t_values) * \
                             basis_normalizations[basis_index] / \
                             estimated_normalizations[estimated_index]
            ax_list[basis_index].plot(t_values, rescaled_basis,
                                      label="true basis %i" % basis_index)

            legend = ax_list[basis_index].legend()
            for label in legend.get_texts():
                label.set_fontsize(12)

    else:
        for estimated_index in range(learner.n_basis):
            estimated = learner.basis_kernels[estimated_index]

            piecewise_y = np.repeat(estimated, 2)
            piecewise_t = np.hstack(
                (learner.kernel_discretization[0],
                 np.repeat(learner.kernel_discretization[1:-1], 2),
                 learner.kernel_discretization[-1]))

            ax_list[estimated_index].plot(
                piecewise_t, piecewise_y,
                label="estimated %i" % estimated_index)
            legend = ax_list[estimated_index].legend()
            for label in legend.get_texts():
                label.set_fontsize(12)

    if show:
        plt.show()

    return fig
