# License: BSD 3 clause
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.benchmark.benchmark_util import (
    iter_executables, run_benchmark, extract_build_from_name,
    default_result_dir, get_last_result_dir)

BASE_FILE_NAME = os.path.basename(__file__).replace('.py', '')


def run_logistic_regression_loss_benchmark():
    result_dir = default_result_dir(base=BASE_FILE_NAME)

    for executable in iter_executables('logistic_regression_loss'):
        n_threads = [1, 2, 4, 6, 8]
        result_dir = run_benchmark(executable, n_threads, result_dir)

    return result_dir


def _load_benchmark_data(result_dir=None):
    if result_dir is None:
        result_dir = get_last_result_dir(BASE_FILE_NAME)

    cols = ["time", "iterations", "n_threads", "n_samples", "n_features",
            "exectuable", "build"]

    df = pd.DataFrame(columns=cols)
    for result_file in [f for f in os.listdir(result_dir)
                        if f.endswith('tsv')]:
        result_path = os.path.join(result_dir, result_file)
        local_df = pd.read_csv(result_path, sep='\t', names=cols[:-1],
                               index_col=False)
        local_df[cols[-1]] = extract_build_from_name(result_file)
        df = df.append(local_df)

    for num_col in [col for col in cols if col not in ['exectuable', 'build']]:
        df[num_col] = pd.to_numeric(df[num_col])

    return df, result_dir


def plot_logistic_regression_loss_benchmark(result_dir=None):
    df, result_dir = _load_benchmark_data(result_dir)

    fig, axes = plt.subplots(1, 2)
    ax_time = axes[0]
    ax_speedup = axes[1]

    max_threads = df['n_threads'].max()
    ax_speedup.plot([1, max_threads], [1, max_threads], linestyle='--', lw=1,
                    color='grey')
    for build, df_build in df.groupby('build'):

        group_by_threads = df_build.groupby('n_threads')
        grouped_times = group_by_threads['time']

        mean_times = grouped_times.mean()
        confidence_times = grouped_times.std() / np.sqrt(grouped_times.count())
        confidence_times *= 1.96

        ax_time.plot(mean_times, label=build)
        ax_time.set_title('Time needed')

        ax_time.fill_between(mean_times.index,
                             mean_times - confidence_times,
                             mean_times + confidence_times,
                             alpha=.3)

        speed_ups = mean_times[1] / mean_times
        ax_speedup.plot(speed_ups, label=build)
        ax_speedup.set_title('Speed up')

        for ax in axes:
            ax.legend()
            ax.set_xlabel('Threads')

    plot_file_path = os.path.abspath(os.path.join(result_dir, 'result.png'))
    plt.savefig(plot_file_path)
    print('saved figure in {}'.format(plot_file_path))


run_logistic_regression_loss_benchmark()
plot_logistic_regression_loss_benchmark()
