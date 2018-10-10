# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np


def plot_point_process(point_process, plot_intensity=None, n_points=10000,
                       plot_nodes=None, node_names=None, t_min=None, t_max=None, 
                       max_jumps=None, show=True, ax=None):
    """Plot point process realization

    Parameters
    ----------
    point_process : `SimuPointProcess`
        Point process that will be plotted

    plot_intensity : `bool`, default=`None`
        Flag saying if intensity should be plotted. If `None`, intensity will
        be plotted if it has been tracked.

    n_points : `int`, default=10000
        Number of points used for intensity plot.

    plot_nodes : `list` of `int`, default=`None`
        List of nodes that will be plotted. If `None`, all nodes are considered

    node_names : `list` of `str`, default=`None`
        List of node names. If `None`, node indices are used.

    t_min : `float`, default=`None`
        If not `None`, time at which plot will start

    t_max : `float`, default=`None`
        If not `None`, time at which plot will stop

    max_jumps : `int`, default=`None`
        If not `None`, maximum of jumps per coordinate that will be plotted.
        This is useful when plotting big point processes to ensure a only
        readable part of them will be plotted

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    ax : `list` of `matplotlib.axes`, default=None
        If not None, the figure will be plot on this axis and show will be
        set to False.
    """
    if plot_nodes is None:
        plot_nodes = range(point_process.n_nodes)

    if node_names is None:
        node_names = list(map(lambda n: 'ticks #{}'.format(n), plot_nodes))
    elif len(node_names) != len(plot_nodes):
        ValueError('node_names must be a list of length {} but has length {}'
                   .format(len(plot_nodes), len(node_names)))
    labels = []
    for name, node in zip(node_names, plot_nodes):
        label = name
        if t_min is not None or t_max is not None:
            if t_min is None:
                label_t_min = '0'
            else:
                label_t_min = '{:.3g}'.format(t_min)
            if t_max is None:
                label_t_max = '{:.3g}'.format(point_process.simulation_time)
            else:
                label_t_max = '{:.3g}'.format(t_max)

            label += ', $t \in [{}, {}]$'.format(label_t_min, label_t_max)

        if max_jumps is not None:
            label += ', max jumps={}'.format(max_jumps)

        labels += [label]

    if ax is None:
        fig, ax = plt.subplots(
            len(plot_nodes), 1, sharex=True, sharey=True,
            figsize=(12, 4 * len(plot_nodes)))
    else:
        show = False

    if len(plot_nodes) == 1:
        ax = [ax]

    if plot_intensity is None:
        plot_intensity = point_process.is_intensity_tracked()

    timestamps = point_process.timestamps
    if plot_intensity:
        intensity_times = point_process.intensity_tracked_times
        intensities = point_process.tracked_intensity
    else:
        intensity_times, intensities = None, None

    timestamps, intensity_times, intensities = _extract_process_interval(
        plot_nodes, point_process.end_time, timestamps,
        intensity_times=intensity_times, intensities=intensities, t_min=t_min,
        t_max=t_max, max_jumps=max_jumps)

    for count, i in enumerate(plot_nodes):
        if not plot_intensity:
            _plot_tick_bars(timestamps[i], ax[count], labels[count])

        else:
            _plot_tick_intensity(timestamps[i], intensity_times,
                                 intensities[i], ax[count], labels[count],
                                 n_points)

    ax[-1].set_xlabel(r'$t$', fontsize=18)

    if show is True:
        plt.show()

    return ax[0].figure


def _plot_tick_bars(timestamps_i, ax, label):
    for t in timestamps_i:
        ax.axvline(x=t)
    ax.set_title(label, fontsize=20)
    ax.get_yaxis().set_visible(False)


def _plot_tick_intensity(timestamps_i, intensity_times, intensity_i, ax, label,
                         n_points):
    x_intensity = np.linspace(intensity_times.min(), intensity_times.max(),
                              n_points)
    y_intensity = np.interp(x_intensity, intensity_times, intensity_i, left=0)
    ax.plot(x_intensity, y_intensity)

    x_ticks = timestamps_i
    y_ticks = np.interp(x_ticks, intensity_times, intensity_i)

    ax.scatter(x_ticks, y_ticks)
    ax.set_title(label)


def _extract_process_interval(plot_nodes, end_time, timestamps,
                              intensities=None, intensity_times=None,
                              t_min=None, t_max=None, max_jumps=None):
    t_min_is_specified = t_min is not None
    if not t_min_is_specified:
        t_min = 0
    t_max_is_specified = t_max is not None
    if not t_max_is_specified:
        t_max = end_time

    if t_min >= end_time:
        raise ValueError('`t_min` should be smaller than `end_time`')
    if t_max <= 0:
        raise ValueError('`t_max` should be positive')

    if max_jumps is not None:
        # if neither t_min or t_ax is given, we act as if t_min=0 was given
        if t_min_is_specified or not t_max_is_specified:
            for i in plot_nodes:
                timestamps_i = timestamps[i]
                i_t_min = np.searchsorted(timestamps_i, t_min, side="left")
                # This might happen if max_points = 0
                last_index = i_t_min + max_jumps - 1
                if last_index < 0:
                    t_max = 0
                elif last_index < len(timestamps_i) \
                        and timestamps_i[last_index] < t_max:
                    t_max = timestamps_i[last_index]

        elif t_max_is_specified:
            for i in plot_nodes:
                timestamps_i = timestamps[i]
                i_t_max = np.searchsorted(timestamps_i, t_max, side="left")
                # This might happen if max_points = 0
                first_index = i_t_max - max_jumps
                if first_index >= len(timestamps_i) - 1:
                    t_min = end_time
                elif first_index >= 0 and timestamps_i[first_index] > t_min:
                    t_min = timestamps_i[first_index]

    extracted_timestamps = [
        timestamps_i[(timestamps_i >= t_min) & (timestamps_i <= t_max)]
        for timestamps_i in timestamps
    ]

    if intensity_times is not None:
        intensity_extracted_points = (intensity_times >= t_min) \
                                     & (intensity_times <= t_max)
        extracted_intensity_times = intensity_times[intensity_extracted_points]

        extracted_intensity = [
            intensity[intensity_extracted_points] for intensity in intensities
        ]
    else:
        extracted_intensity_times, extracted_intensity = None, None

    return extracted_timestamps, extracted_intensity_times, extracted_intensity
