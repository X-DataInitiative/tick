# md stands for multiple dimensions
import itertools

import numpy as np
import matplotlib.pyplot as plt

from experiments.dict_utils import nested_update
from experiments.grid_search_1d import get_best_point
from experiments.metrics_utils import strength_range_from_infos
from experiments.toy_metrics import get_toy_metrics_2d


def get_new_range_2d(infos, prox_info, metrics,
                     extension_step=1e3, limit_low=1e-10, limit_high=1e2):
    def add_between(p1, p2, p_list):
        if max(p1 / p2, p2 / p1) > prox_info['max_relative_step']:
            p_list += [np.sqrt(p1 * p2)]

    strength_range_md = strength_range_from_infos(infos)
    # number of parameters
    dim = len(strength_range_md[0])
    dims = range(dim)

    current_min_range = {}
    current_max_range = {}
    dim_strength_ranges = {}
    for d in dims:
        dim_strength_range = np.array(
            list(set([strength[d] for strength in strength_range_md])))
        dim_strength_range.sort()
        dim_strength_ranges[d] = dim_strength_range
        current_min_range[d] = min(dim_strength_range[dim_strength_range != 0])
        current_max_range[d] = max(dim_strength_range)

    new_strength_range = set()
    for metric in metrics:
        best_point = get_best_point(metrics, metric, infos)
        best_point_md = {}
        for d in dims:
            dim_strength_range = dim_strength_ranges[d]
            dim_best_point = \
                np.where(
                    dim_strength_range == strength_range_md[best_point][d])[
                    0][0]
            best_point_md[d] = dim_best_point

        best_point_types = {}
        for d in dims:
            dim_strength_range = dim_strength_ranges[d]
            dim_best_point = best_point_md[d]
            if dim_best_point == 0 or (
                    dim_best_point == 1 and dim_strength_range[0] == 0):
                if current_min_range[d] > 1e-13:
                    best_point_types[d] = 'border low'
                else:
                    best_point_types[d] = 'limit low'
            elif dim_best_point == len(dim_strength_range) - 1:
                if current_max_range[d] < 1e5:
                    best_point_types[d] = 'border high'
                else:
                    best_point_types[d] = 'limit high'
            else:
                best_point_types[d] = 'inside'

        # extension style
        if np.any(np.array(['border' in best_point_types[d] for d in dims])):
            extension_points = {}
            for d in dims:
                dim_strength_range = dim_strength_ranges[d]
                dim_best_point = best_point_md[d]
                extension_points[d] = [
                    dim_strength_range[dim_best_point]]  # best
                if best_point_types[d] == 'border low':
                    extension_points[d] += [
                        current_min_range[d] / extension_step]  # smaller
                    extension_points[d] += [
                        dim_strength_range[dim_best_point + 1]]  # bigger
                elif best_point_types[d] == 'border high':
                    extension_points[d] += [
                        dim_strength_range[dim_best_point - 1]]  # smaller
                    extension_points[d] += [
                        current_max_range[d] * extension_step]  # bigger
                elif best_point_types[d] == 'limit low':
                    lowest_acceptable_point = min(
                        dim_strength_range[dim_strength_range > limit_low])
                    extension_points[d] += [
                        np.sqrt(limit_low * lowest_acceptable_point)]
                elif best_point_types[d] == 'limit high':
                    bigger_acceptable_point = max(
                        dim_strength_range[dim_strength_range < limit_high])
                    extension_points[d] += [
                        np.sqrt(limit_high * bigger_acceptable_point)]
                elif best_point_types[d] == 'inside':
                    extension_points[d] += [
                        dim_strength_range[dim_best_point - 1]]  # smaller
                    extension_points[d] += [
                        dim_strength_range[dim_best_point + 1]]  # bigger
                else:
                    raise (ValueError('Unknown type'))

            for new_point in itertools.product(
                    *[extension_points[d] for d in dims]):
                if new_point not in infos[0]['alpha_auc']:
                    new_strength_range.add(new_point)
        # precision style
        else:
            precision_points = {}
            for d in dims:
                dim_strength_range = dim_strength_ranges[d]
                dim_best_point = best_point_md[d]
                precision_points[d] = [
                    dim_strength_range[dim_best_point]]  # best
                if best_point_types[d] == 'limit low':
                    lowest_acceptable_point = min(
                        dim_strength_range[dim_strength_range > limit_low])
                    add_between(limit_low, lowest_acceptable_point,
                                precision_points[d])
                elif best_point_types[d] == 'limit high':
                    bigger_acceptable_point = max(
                        dim_strength_range[dim_strength_range < limit_high])
                    add_between(limit_low, bigger_acceptable_point,
                                precision_points[d])
                elif best_point_types[d] == 'inside':
                    add_between(dim_strength_range[dim_best_point - 1],
                                dim_strength_range[dim_best_point],
                                precision_points[d])  # smaller
                    add_between(dim_strength_range[dim_best_point + 1],
                                dim_strength_range[dim_best_point],
                                precision_points[d])  # bigger
                else:
                    raise (ValueError('Unknown type'))

            for new_point in itertools.product(
                    *[precision_points[d] for d in dims]):
                if new_point not in infos[0]['alpha_auc']:
                    new_strength_range.add(new_point)

    new_strength_range = list(new_strength_range)
    new_strength_range.sort()
    return new_strength_range


def geometrical_center(axis_values):
    middle_values = [np.sqrt(l * u) for (l, u) in
                     zip(axis_values[:-1], axis_values[1:])]
    first_value = axis_values[0] ** 2 / middle_values[0]
    last_value = axis_values[-1] ** 2 / middle_values[-1]
    axis_centered = [first_value] + middle_values + [last_value]
    return np.array(axis_centered)


def plot_2d_metric(infos, metric_name, best, ax=None):
    points = []
    z = []
    for k, v in infos[0][metric_name].items():
        points += [k]
        z += [v]

    points = np.array(points)
    z = np.array(z)

    grid = 50
    log_xi = np.linspace(-15, 5, grid)
    log_yi = np.linspace(-15, 5, grid)
    log_xi, log_yi = np.meshgrid(log_xi, log_yi)
    log_xi_flat = log_xi.reshape(grid * grid)
    log_yi_flat = log_yi.reshape(grid * grid)

    space = np.array([(xj, yj) for xj, yj in zip(log_xi_flat, log_yi_flat)])
    import scipy
    mytree = scipy.spatial.cKDTree(np.log10(points))
    dist_i, closest_i = mytree.query(space)
    zi = z[closest_i]
    # if the closest point is at more than a factor 10^2 we remove its value
    zi[dist_i > 2] = np.nan
    zi = zi.reshape((grid, grid))

    xi = np.power(10, log_xi)
    yi = np.power(10, log_yi)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    if best == 'min':
        cmap = 'Reds_r'
    elif best == 'max':
        cmap = 'Reds'
    CS = ax.contourf(xi, yi, zi, 15, cmap=cmap)
    # cbar = plt.colorbar(CS)
    ax.set_xscale('symlog', linthreshx=1e-15)
    ax.set_yscale('symlog', linthreshy=1e-15)
    ax.set_title(metric_name)


def plot_all_2d_metrics(infos, metrics, ax=None):
    if ax is None:
        _, ax = plt.subplots(2, 3)
    fig = ax.ravel()[0].get_figure()
    for i, metric in enumerate(metrics.keys()):
        plot_2d_metric(infos, metric, metrics[metric]['best'],
                       ax=ax.ravel()[i])
    fig.tight_layout()
    return ax, fig


if __name__ == '__main__':
    from experiments.tested_prox import create_prox_l1_no_mu_nuclear

    toy_metrics_2d = get_toy_metrics_2d()

    def toy_learn2d(toy_strength_range):
        toy_infos = {}
        for metric in toy_metrics_2d:
            for i in range(3):
                if i not in toy_infos:
                    toy_infos[i] = {}
                metric_values = {
                    strength: toy_metrics_2d[metric]['evaluator'](
                        *strength) + np.random.uniform(-0.0001, 0.0001)
                    for strength in toy_strength_range}
                toy_infos[i][metric] = metric_values
        return toy_infos


    prox_info = {
        'name': 'l1_nuclear',
        'n_initial_points': 10,
        'max_relative_step': 5,
        'create_prox': create_prox_l1_no_mu_nuclear,
        'tol': 1e-8,
        'dim': 2,
    }

    toy_strength_range_1 = np.logspace(-8, -2, 3)
    toy_strength_range_2 = np.logspace(-8, -2, 3)
    toy_strength_range_l1_nuclear = [
        (l1, tau) for (l1, tau)
        in itertools.product(toy_strength_range_1, toy_strength_range_2)
    ]

    aggregated_infos = dict()
    for run_count in range(10):

        toy_infos_2d = toy_learn2d(toy_strength_range_l1_nuclear)
        aggregated_infos = nested_update(toy_infos_2d, aggregated_infos)

        plot_2d_metric(toy_infos_2d, 'alpha_auc', 'max')
        plt.show()
        print('Run %i' % (run_count))
        toy_strength_range_l1_nuclear = get_new_range_2d(
            toy_infos_2d, prox_info, toy_metrics_2d, extension_step=1e2)
        print(len(toy_strength_range_l1_nuclear))
        if len(toy_strength_range_l1_nuclear) == 0:
            break

    plot_all_2d_metrics(aggregated_infos, toy_metrics_2d)
    plt.show()
