import numpy as np


def stem(y: np.ndarray, show=True, title=None, x_range=None, y_range=None):
    """Stem plot using bokeh

    Parameters
    ----------
    y : `np.ndarray`, shape=(n_coeffs,)
        The vector to plot

    show : `bool`, default=`True`
        if True, show the plot. Use False if you want to superpose several
        plots

    title : `str`, default=`None`
        The title of the plot

    Returns
    -------
    output : `bk.figure`
        A bokeh figure object
    """
    import bokeh.plotting as bk

    dim = y.shape[0]
    x = np.arange(dim)
    plot_width = 600
    plot_height = 200
    fig = bk.figure(plot_width=plot_width, plot_height=plot_height,
                    x_range=x_range, y_range=y_range)
    fig.scatter(x, y, size=4, fill_alpha=0.5)
    fig.segment(x, np.zeros(dim), x, y)
    fig.title_text_font_size = "12pt"
    if title is not None:
        fig.title = title
    if show:
        bk.show(fig)
        return None
    else:
        return fig


def stems(ys: list, titles: list = None, show=True, sync_axes=True):
    """Several stem plots with synchronized axes

    Parameters
    ----------
    ys : `list` of `np.ndarray`
        A list of numpy arrays to be plotted

    titles : `list` of `str`
        The titles of each plot

    show : `bool`, default=`True`
        if True, show the plot. Use False if you want to superpose several
        plots

    sync_axes : `bool`, default=`True`
        If True, the axes of the stem plot are synchronized

    Returns
    -------
    output : `bk.grid_plot`
        A grid plot object
    """
    import bokeh.plotting as bk

    figs = []
    x_range = None
    y_range = None
    for idx, y in enumerate(ys):
        if titles is not None:
            title = titles[idx]
        else:
            title = None
        fig = stem(y, show=False, title=title, x_range=x_range, y_range=y_range)
        figs.append(fig)
        if idx == 0 and sync_axes:
            x_range = fig.x_range
            y_range = fig.y_range
    p = bk.gridplot([[e] for e in figs])
    if show:
        bk.show(p)
        return None
    else:
        return p
