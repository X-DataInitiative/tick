import os
import time
import uuid

import pandas as pd

from experiments.grid_search import get_best_point, plot_all_metrics
from experiments.metrics import get_metrics
from experiments.metrics_utils import mean_and_std, extract_metric, \
    strength_range_from_infos

MEAN_PREFIX = 'metrics_mean'
STD_PREFIX = 'metrics_std'
LAMBDAS_PREFIX = 'used_lambdas'


def get_csv_path(prefix, suffix):
    return "{}_{}.csv".format(prefix, suffix)


def read_csv(prefix, suffix, metrics):
    file_path = get_csv_path(prefix, suffix)
    if os.path.exists(file_path):
        return pd.read_csv()
    else:
        columns = ['dim', 'prox', 'end_time', 'n_train'] + list(metrics.keys())
        return pd.DataFrame(columns)


def read_means_csv(suffix, metrics):
    return read_csv(MEAN_PREFIX, suffix, metrics)


def write_means_csv(suffix, df):
    df.to_csv(get_csv_path(MEAN_PREFIX, suffix) % suffix, index=False)


def read_stds_csv(suffix, metrics):
    return read_csv(STD_PREFIX, suffix, metrics)


def write_stds_csv(suffix, df):
    df.to_csv(get_csv_path(STD_PREFIX, suffix) % suffix, index=False)


def read_lambdas_csv(suffix, metrics):
    return read_csv(LAMBDAS_PREFIX, suffix, metrics)


def write_lambdas_csv(suffix, df):
    df.to_csv(get_csv_path(LAMBDAS_PREFIX, suffix) % suffix, index=False)


def save_best_metrics(suffix, metrics, infos, dim, run_time, n_trains,
                     prox_name):
    best_metrics_mean = read_means_csv(suffix, metrics)
    best_metrics_std = read_stds_csv(suffix, metrics)
    best_lambdas = read_lambdas_csv(suffix, metrics)

    run_best_metrics_mean = {
        'dim': dim,
        'prox': prox_name,
        'T': run_time,
        'n_train': n_trains,
    }
    run_best_lambda = run_best_metrics_mean.copy()
    run_best_metrics_std = run_best_metrics_mean.copy()

    for metric in metrics:
        strength_range = strength_range_from_infos(infos)
        metric_metrics = extract_metric(metric, infos)
        mean_metrics, std = mean_and_std(metric_metrics)
        best_point = get_best_point(metrics, metric, infos)

        run_best_metrics_mean[metric] = mean_metrics[best_point]
        run_best_metrics_std[metric] = std[best_point]
        run_best_lambda[metric] = strength_range[best_point]

    best_metrics_mean = best_metrics_mean.append(run_best_metrics_mean,
                                                 ignore_index=True)
    best_metrics_std = best_metrics_std.append(run_best_metrics_std,
                                               ignore_index=True)
    best_lambdas = best_lambdas.append(run_best_lambda, ignore_index=True)

    write_means_csv(suffix, best_metrics_mean)
    write_stds_csv(suffix, best_metrics_std)
    write_lambdas_csv(suffix, best_lambdas)


def get_image_directory(dim, run_time, prox_name):
    dir_name = 'dim_{}/T_{}/prox_{}'.format(dim, run_time, prox_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def record_metrics(infos, dim, run_time, n_trains, prox_name, logger, suffix):
    dir_name = get_image_directory(dim, run_time, prox_name)
    ax, fig = plot_all_metrics(infos, get_metrics())

    graph_file_path = os.path.join(
        dir_name, 'run_{}_{}.png'.format(int(time.time()), str(uuid.uuid4())))

    fig.savefig(graph_file_path, dpi=100, format='png', bbox_inches='tight')
    logger('![alt text](%s "Title")' % graph_file_path)

    save_best_metrics(suffix, get_metrics(), infos, dim, run_time, n_trains,
                      prox_name)

