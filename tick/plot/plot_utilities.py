import matplotlib.pyplot as plt
import numpy as np


def get_plot_color(value, n_values=11, palette='Set3'):
    try:
        from bokeh.palettes import brewer
        palette_set = brewer['Spectral']
        palette = palette_set[max(min(n_values, len(palette_set)), 3)]
        return palette[value]

    except ImportError:

        import matplotlib.cm as cmx
        import matplotlib.colors as colors

        cm = plt.get_cmap(palette)
        c_norm = colors.Normalize(vmin=0, vmax=1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        return scalar_map.to_rgba(value / n_values)


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
