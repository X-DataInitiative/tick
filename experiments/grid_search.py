# License: BSD 3 clause
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from experiments.dict_utils import nested_update
from experiments.metrics_utils import (
    extract_metric, mean_and_std, strength_range_from_infos,
    get_confidence_interval_half_width)


def get_best_point(metrics, metric, infos):
    metric_values = extract_metric(metric, infos)
    mean, _ = mean_and_std(metric_values)
    metric_info = metrics[metric]
    if metric_info['best'] == 'min':
        best_point = np.nanargmin(mean)
    elif metric_info['best'] == 'max':
        best_point = np.nanargmax(mean)
    else:
        raise ValueError('Unknown best metric')
    return best_point


def get_new_range(infos, prox_name, metrics, prox_infos):
    strength_range = strength_range_from_infos(infos)

    current_min_range = min(strength_range[strength_range != 0])
    current_max_range = max(strength_range)

    new_strength_range = []
    for metric in metrics:
        best_point = get_best_point(metrics, metric, infos)

        if best_point == 0 or (best_point == 1 and strength_range[0] == 0):
            if current_min_range > 1e-13:
                extension_point = current_min_range * 1e-3
                new_strength_range += list(
                    np.logspace(np.log10(extension_point),
                                np.log10(current_min_range), 4))

        elif best_point == len(strength_range) - 1:
            if current_max_range < 1e5:
                extension_point = current_max_range * 1e3
                new_strength_range += list(
                    np.logspace(np.log10(current_max_range),
                                np.log10(extension_point), 4))

        else:
            previous_best_strength = strength_range[best_point - 1]
            best_strength = strength_range[best_point]
            next_best_strength = strength_range[best_point + 1]

            # If we are not close enough from the best point
            if next_best_strength / previous_best_strength > \
                    prox_infos[prox_name]['max_relative_step']:
                new_strength_range += [
                    np.sqrt(previous_best_strength * best_strength)]
                new_strength_range += [
                    np.sqrt(next_best_strength * best_strength)]

    new_strength_range = np.array(list(set(new_strength_range)))
    new_strength_range.sort()
    return new_strength_range


def plot_with_errorbars(data_array, x_axis, ax, label='', loc='best',
                        best='min'):
    """
    data_array is an array has on each line the value of the loss for one
    model for all strength of strength range
    """
    mean, std = mean_and_std(data_array)
    ci_hw = get_confidence_interval_half_width(std, data_array.shape[0])

    if best == 'min':
        best_point = np.nanargmin(mean)
    elif best == 'max':
        best_point = np.nanargmax(mean)

    ax.axvline(x_axis[best_point], color='red', lw=3)
    ax.text(x_axis[best_point], min(mean), '%.3g\n%.3g +/- %.3g' %
            (x_axis[best_point], mean[best_point], ci_hw[best_point]),
            fontsize=16)

    ax.errorbar(x_axis, mean, yerr=ci_hw, label=label)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xscale('symlog', linthreshx=min(x_axis[x_axis != 0]) * 1e-2)


def plot_all_metrics(infos, metrics, ax=None):
    strength_range = strength_range_from_infos(infos)
    if ax is None:
        n_plots = len(metrics.keys())
        fig, ax = plt.subplots(n_plots, 1, sharex=True)

    for i, metric in enumerate(metrics.keys()):
        metric_values = extract_metric(metric, infos)
        plot_with_errorbars(metric_values, strength_range, ax[i], label=metric,
                            loc='best', best=metrics[metric]['best'])

    # we return axes in case you want to modify them afterwards
    return ax, fig


if __name__ == '__main__':
    def toy_auc(x):
        if x < 1e-5: return toy_auc(1e-5) * 0.99
        if x > 1e2: return toy_auc(1e2) * 1.01
        return 0.5 - 0.1 * np.arctan(5 * np.log10(x / 1e-3))


    def toy_kendall(x):
        if x == 0:
            return toy_kendall(1e-100)
        a = np.exp(-3 * np.abs(np.log10(x / 1e-3)))
        b = np.cos(np.log10(x / 1e-3))
        return a * b


    def toy_kendall_no_diag(x):
        if x > 1e-1:
            return np.nan
        return toy_kendall(x * 1e3)


    toy_auc = np.vectorize(toy_auc)
    xaxis = np.hstack((0, np.logspace(-9, 4)))
    yaxis = toy_auc(xaxis)
    ax = plt.subplot(211)
    ax.plot(xaxis, yaxis)
    ax.set_xscale('symlog', linthreshx=1e-10)
    ax.set_title('toy_auc')

    toy_kendall = np.vectorize(toy_kendall)

    xaxis = np.hstack((0, np.logspace(-9, 4)))
    yaxis = toy_kendall(xaxis)
    ax = plt.subplot(212)
    ax.plot(xaxis, yaxis)
    ax.set_xscale('symlog', linthreshx=1e-10)
    ax.set_title('toy_kendall')

    toy_estimation_error = toy_auc

    plt.show()

    toy_metrics = OrderedDict()
    toy_metrics["alpha_auc"] = {'evaluator': toy_auc, 'best': 'max'}
    toy_metrics["estimation_error"] = {'evaluator': toy_auc, 'best': 'min'}
    toy_metrics["kendall"] = {'evaluator': toy_kendall, 'best': 'max'}
    toy_metrics["kendall_no_diag"] = {
        'evaluator': lambda x: toy_kendall(x * 1e-3),
        'best': 'max'
    }


    def toy_learn(toy_strength_range, n_models):
        toy_infos = {}
        for metric in toy_metrics:
            for i in range(n_models):
                if i not in toy_infos:
                    toy_infos[i] = {}
                metric_values = {strength: toy_metrics[metric]['evaluator'](
                    strength) + np.random.uniform(-0.0001, 0.0001)
                                 for strength in toy_strength_range}
                toy_infos[i][metric] = metric_values
        return toy_infos


    toy_strength_range = np.hstack((0, np.logspace(-5, -2, 3)))
    toy_infos = {}

    prox_infos = {}
    prox_infos['l1'] = {
        'n_initial_points': 10,
        'max_relative_step': 1.4,
        'tol': 1e-10,
        'dim': 1,
    }

    for run_count in range(10):
        print('Run %i' % (run_count))

        old_toy_infos = toy_infos

        toy_infos = toy_learn(toy_strength_range, 3)
        toy_infos = nested_update(toy_infos, old_toy_infos)
        toy_strength_range = get_new_range(toy_infos, 'l1', toy_metrics,
                                           prox_infos)
        if len(toy_strength_range) == 0:
            break

    plot_all_metrics(toy_infos, toy_metrics)
    plt.show()
