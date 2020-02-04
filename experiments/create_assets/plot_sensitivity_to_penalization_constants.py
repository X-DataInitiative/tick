import matplotlib.pyplot as plt
import numpy as np

from experiments.sensitivity_to_regularization_constant import get_range_for_metric, plot_metrics_for_strength_range


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
                # ax[i, j].get_yaxis().set_ticks([])
                pass


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

def format_sub_title(prox_name, metric):

    if prox_name == 'l1':
        prox_label = 'L1'
    else:
        prox_label = 'wL1'

    if metric == 'alpha_auc':
        metric_label = 'AUC'
    elif metric == 'estimation_error':
        metric_label = 'Estimation error'
    else:
        metric_label = 'Kendall'

    return f'{prox_label} - {metric_label}'


if __name__ == "__main__":
    dim = 30
    run_time = 10000
    training_suffix = '100_x_25000_iter'
    suffix = 'width_10'
    metrics = ['alpha_auc', 'estimation_error', 'kendall']

    share_axes = False
    fig, axes = plt.subplots(2, 3, figsize=(10, 4))
    for i, prox_name in enumerate(['l1', 'l1w']):
        for j, metric in enumerate(metrics):
            prox_info = {'name': prox_name}
            ax = axes[i, j]
            strength_range = get_range_for_metric(dim, run_time, prox_info, metric, training_suffix=training_suffix)
            plot_metrics_for_strength_range(metric, prox_info, suffix=suffix, ax=ax, strength_range=strength_range)
            ax.set_title(format_sub_title(prox_name, metric))

    if share_axes:
        share_x(axes)
        for j in range(len(metrics)):
            share_y(axes[:, j].reshape(1, -1))
            # axes[1, j].get_yaxis().set_ticks(axes[1, j].get_yticks())
            y_inf, y_sup = axes[0, j].get_ylim()
            delta_y = (y_sup - y_inf) / 10
            axes[0, j].set_ylim([y_inf - delta_y, y_sup + delta_y])
            axes[1, j].set_ylim([y_inf - delta_y, y_sup + delta_y])


    fig_name = f'../figs/sensitivity_to_lambda{"_share_axes" if share_axes else ""}.pdf'
    plt.savefig(fig_name)
    # plt.show()