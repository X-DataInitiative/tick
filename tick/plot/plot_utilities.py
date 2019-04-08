# License: BSD 3 clause

import numpy as np

# Matplotlib colors of tab cmap (previously called Vega)
# It has been re-ordered so that light colors apperas at the end
tab20_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
    '#dbdb8d', '#9edae5'
]


def get_plot_color(number):
    """Return color for a line number.

    Color are extracted from tab20 colormap which is an extension of
    matplotlib 2.x CN colors. 20 colors are available.

    Parameters
    ----------
    number : `int`
        Number of the color to pick

    Returns
    -------
    color : `str`
        Color in hexadecimal format
    """
    return tab20_colors[number % len(tab20_colors)]


def share_y(ax):
    """Manually share y axis on an array of axis
    
    Parameters
    ----------
    ax : `np.ndarray`
        2d array of axes that will share y
        
    Notes
    -----
    This utlity is useful as sharey kwarg of subplots cannot be applied only 
    on a subset of the axes
    """
    n_rows, n_cols = ax.shape
    get_ylim = np.vectorize(lambda axis: axis.get_ylim())
    y_min, y_max = get_ylim(ax)
    y_min_min = y_min.min()
    y_max_max = y_max.max()
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].set_ylim([y_min_min, y_max_max])
            if j != 0:
                ax[i, j].get_yaxis().set_ticks([])


def share_x(ax):
    """Manually share x axis on an array of axis

    Parameters
    ----------
    ax : `np.ndarray`
        2d array of axes that will share x

    Notes
    -----
    This utlity is useful as sharex kwarg of subplots cannot be applied only 
    on a subset of the axes
    """
    n_rows, n_cols = ax.shape
    get_xlim = np.vectorize(lambda axis: axis.get_xlim())
    x_min, x_max = get_xlim(ax)
    x_min_min = x_min.min()
    x_max_max = x_max.max()
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].set_xlim([x_min_min, x_max_max])
            if i != n_rows - 1:
                ax[i, j].get_xaxis().set_ticks([])
