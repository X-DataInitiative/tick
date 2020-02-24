import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.dict_utils import nested_update
from experiments.metrics_utils import (
    strength_range_from_infos, extract_metric, mean_and_std,
    get_confidence_interval_half_width
)

try:
    from experiments.learning import learn_one_strength_range
    from experiments.weights_computation import load_models, LEAST_SQ_LOSS
except ImportError:
    print('tick not correctly installed')

from experiments.report_utils import read_json, write_json, LAMBDAS_PREFIX, get_csv_path


def get_sensitivity_infos_prefix(prox_info):
    return f'sensitivity_infos_{prox_info["name"]}'


def get_range_for_metric(dim, run_time, prox_info, metric, n_train=100, training_suffix='', width=10, n_points=10):
    file_path = get_csv_path(LAMBDAS_PREFIX, training_suffix)
    best_lambdas_df = pd.read_csv(file_path)
    mask = best_lambdas_df['dim'] == dim
    mask &= best_lambdas_df['end_time'] == run_time
    mask &= best_lambdas_df['n_train'] == n_train
    mask &= best_lambdas_df['prox'] == prox_info['name']

    best_lambda = float(best_lambdas_df[mask][metric].values[0])

    min_lambda, max_lambda = best_lambda / width, best_lambda * width
    lambda_range = np.logspace(np.log10(min_lambda), np.log10(max_lambda), n_points)
    lambda_range = set(lambda_range)
    lambda_range.add(best_lambda)
    return sorted(list(lambda_range))


def get_metrics_for_strength_range(strength_range, dim, run_time, n_decays, n_models, prox_info,
                                   solver_kwargs, directory_path, n_cpu=-1, suffix='', loss=LEAST_SQ_LOSS):
    original_coeffs, model_file_paths = load_models(dim, run_time, n_decays, n_models, directory_path, loss)

    infos = learn_one_strength_range(original_coeffs, model_file_paths, strength_range,
                                     prox_info, solver_kwargs, n_cpu)

    prefix = get_sensitivity_infos_prefix(prox_info)
    existing_infos = read_json(prefix, suffix)
    new_infos = nested_update(existing_infos, infos)
    write_json(prefix, suffix, new_infos)


def plot_metrics_for_strength_range(metric, prox_info, suffix='', ax=None, strength_range=None):
    infos = read_json(get_sensitivity_infos_prefix(prox_info), suffix)

    all_strength_range = strength_range_from_infos(infos)
    all_strength_range = np.array(all_strength_range)
    mask = np.isin(all_strength_range, strength_range)

    x_axis = np.array(strength_range)

    metric_values = extract_metric(metric, infos)
    mean, std = mean_and_std(metric_values)
    ci_hw = get_confidence_interval_half_width(std, metric_values.shape[0])

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.errorbar(x_axis, mean[mask], yerr=ci_hw[mask])
    ax.set_xscale('symlog', linthreshx=min(x_axis[x_axis != 0]) * 1e-2)
