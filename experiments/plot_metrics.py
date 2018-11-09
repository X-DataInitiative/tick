import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from experiments.metrics_utils import get_confidence_interval_half_width

prox_infos = OrderedDict()
# prox_infos['l1w'] = 'C1', 'wL1 norm'
prox_infos['l1w_un'] = 'C4', 'wL1'
prox_infos['l1'] = 'C0', 'L1'
# prox_infos['l1w_nuclear'] = 'C3', 'wL1Nuclear norm'
prox_infos['l1w_un_nuclear'] = 'C5', 'wL1Nuclear'
prox_infos['l1_nuclear'] = 'C2', 'L1Nuclear'


def plot_metrics_line(dim, metrics, mean_file, std_file,
                      prox_names, ax=None):
    best_values_mean = pd.read_csv(mean_file)
    best_values_std = pd.read_csv(std_file)

    if ax is None:
        _, ax = plt.subplots(1, len(metrics), figsize=(8, 3))

    for i, metric in enumerate(metrics):
        for prox_name in prox_names:
            line_filer = (best_values_mean['prox'] == prox_name) & \
                         (best_values_mean['dim'] == dim)

            prox_values_mean = best_values_mean[line_filer]
            prox_values_std = best_values_std[line_filer]

            means = prox_values_mean[metric]
            stds = prox_values_std[metric]
            n_trains = prox_values_std['n_train']

            hw_cis = get_confidence_interval_half_width(
                stds, n_trains, confidence=0.95)

            times = prox_values_mean['end_time']

            color, label = prox_infos[prox_name]

            if 'wei' not in prox_name:
                ls = '-.'
            else:
                ls = ':'

            ax[i].fill_between(times, means + hw_cis, means - hw_cis,
                               alpha=0.3, color=color)
            ax[i].plot(times, means, color=color, label=label, ls=ls)

        if 'auc' in metric.lower():
            ax[i].set_ylim([None, 1])

#        if 'error' in metric.lower():
#            ax[i].set_yscale('log')


def plot_metrics(dim, mean_file, std_file, with_diag=True):
    if with_diag:
        metrics = ['alpha_auc', 'estimation_error', 'kendall']
    else:
        metrics = ['alphas_auc_no_diag', 'estimation_error_no_diag',
                   'kendall_no_diag']

    fig, axes = plt.subplots(2, len(metrics), figsize=(8, 4.5),
                             sharex=True)

    prox_no_nuclear = [p for p in prox_infos.keys() if 'nuclear' not in p]
    plot_metrics_line(dim, metrics, mean_file, std_file,
                      prox_names=prox_no_nuclear, ax=axes[0, :])

    prox_nuclear = [p for p in prox_infos.keys() if 'nuclear' in p]
    plot_metrics_line(dim, metrics, mean_file, std_file,
                      prox_names=prox_nuclear, ax=axes[1, :])

    legends = []
    for ax in axes[:, -1]:
        legends += [ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))]

    for ax, title in zip(axes[0, :],
                         ['AUC', 'Estimation error', 'Kendall']):
        ax.set_title(title, fontsize=15)

    for ax in axes[-1, :]:
        ax.set_xlabel('$T$', fontsize=12)

    return fig, legends


if __name__ == '__main__':

    mean_file = 'metrics_mean_test.csv'
    std_file = 'metrics_std_test.csv'

    for dim in [30]:
        fig, legends = plot_metrics(dim, mean_file, std_file, with_diag=False)
        fig.tight_layout(pad=3)
        plt.show()
        # fig.savefig('metrics_dim_{}.pdf'.format(dim), bbox_inches='tight',
        #             bbox_extra_artists=(*legends, ))

