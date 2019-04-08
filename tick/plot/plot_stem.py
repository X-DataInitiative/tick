# License: BSD 3 clause

import numpy
from matplotlib.pyplot import Axes


def __stem_matplotlib(y: numpy.array, axis: Axes, title: str, x_range: tuple,
                      y_range: tuple):
    axis.stem(y, title=title)
    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)
    if title is not None:
        axis.set_title(title, fontsize=16)
    return axis


def __stems_matplotlib(ys: list, titles: list, x_range: tuple, y_range: tuple,
                       fig_size):
    from matplotlib.pyplot import figure

    n_stems = len(ys)
    fig = figure(figsize=(fig_size[0], fig_size[1] * n_stems))
    for idx, (y, title) in enumerate(zip(ys, titles)):
        ax = fig.add_subplot(n_stems, 1, idx + 1)
        __stem_matplotlib(y, ax, title, x_range, y_range)
    return fig


def __stem_bokeh(y: numpy.array, title, x_range, y_range, fig_size):
    import numpy as np
    from bokeh.plotting import figure

    dim = y.shape[0]
    x = np.arange(dim)
    plot_width, plot_height = fig_size
    fig = figure(plot_width=plot_width, plot_height=plot_height,
                 x_range=x_range, y_range=y_range)
    fig.scatter(x, y, size=4, fill_alpha=0.5)
    fig.segment(x, np.zeros(dim), x, y)
    fig.title.text_font_size = "12pt"
    if title is not None:
        fig.title.text = title
    return fig


def __stems_bokeh(ys: list, titles: list, sync_axes, fig_size):
    from bokeh.plotting import gridplot

    figs = []
    x_range = None
    y_range = None
    for idx, (y, title) in enumerate(zip(ys, titles)):
        fig = __stem_bokeh(y, title=title, x_range=x_range, y_range=y_range,
                           fig_size=fig_size)
        figs.append(fig)
        if idx == 0 and sync_axes:
            x_range = fig.x_range
            y_range = fig.y_range
    return gridplot([[e] for e in figs])


def stems(ys: list, titles: list = None, sync_axes: bool = True,
          rendering: str = 'matplotlib', fig_size: tuple = None):
    """Plot several stem plots using either matplotlib or bokeh rendering.
    Axes can be synchronized, which means that all xlim and ylim are the
    same for matplotlib rendering, while zooming is done simultaneously on all
    stem plots for bokeh rendering.

    Parameters
    ----------
    ys : `list` of `numpy.array`
        A list of numpy arrays to be plotted using stem plots

    titles : `list` of `str`, default=`None`
        The titles of each stem plot

    sync_axes : `bool`, default=`True`
        If `True`, axes can be synchronized, which means that for
        matplotlib rendering, the
        xlim and ylim are the same, while for bokeh rendering, this means that
        zooming is done simultaneously on all stems.

        If True, the axes of the stem plot are synchronized (available only
        with ``rendering='bokeh'``

    rendering : {'matplotlib', 'bokeh'}, default='matplotlib'
        Rendering library. 'bokeh' might fail if the module is not installed.

    fig_size: `tuple`, default=`None`
        Figure size. Default is (8, 2.5) for matplotlib rendering and
        (600, 200) for bokeh rendering.

    Returns
    -------
    output : `bokeh.models.layouts.Column`, `matplotlib.pyplot.Figure` or `None`
        Depending on the rendering type, returns a ``bokeh`` or a ``matplotlib``
        object containing the layout of the plot, or `None` otherwise
    """
    if titles is not None:
        if len(ys) != len(titles):
            raise ValueError('length of ``titles`` differs from the length of '
                             '``ys``')
    else:
        titles = len(ys) * [None]

    if rendering == 'matplotlib':
        if fig_size is None:
            fig_size = (8, 2.5)
        if sync_axes:
            x_min = 0
            x_max = max(y.shape[0] for y in ys)
            y_min = min(y.min() for y in ys)
            y_max = max(y.max() for y in ys)
            y_min *= 1 - 5e-2
            y_max *= 1 + 5e-2
            x_range = (x_min, x_max)
            y_range = (y_min, y_max)
        else:
            x_range = None
            y_range = None
        fig = __stems_matplotlib(ys=ys, titles=titles, x_range=x_range,
                                 y_range=y_range, fig_size=fig_size)
        fig.tight_layout()
    elif rendering == 'bokeh':
        from bokeh.plotting import show as bk_show

        if fig_size is None:
            fig_size = (600, 200)
        fig = __stems_bokeh(ys=ys, titles=titles, sync_axes=sync_axes,
                            fig_size=fig_size)
        bk_show(fig)
    else:
        raise ValueError("Unknown rendering type. Expected 'matplotlib' or "
                         "'bokeh', got '%s'" % rendering)
