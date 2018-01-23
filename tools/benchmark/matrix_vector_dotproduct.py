# License: BSD 3 clause
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.benchmark.benchmark_util import (
    iter_executables, run_benchmark, default_result_dir, get_last_result_dir,
    extract_build_from_name)

BASE_FILE_NAME = os.path.basename(__file__).replace('.py', '')


def run_matrix_vector_dotproduct_benchmarks():
    result_dir = default_result_dir(base=BASE_FILE_NAME)

    for executable in iter_executables('matrix_vector_dotproduct'):
        run_benchmark(executable, [None], result_dir)

    return result_dir


def _load_benchmark_data(result_dir=None):
    if result_dir is None:
        result_dir = get_last_result_dir(BASE_FILE_NAME)

    cols = ["time", "iterations", "n_rows", "n_cols", "exectuable", "build"]

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


def plot_matrix_vector_dotproduct_benchmark(result_dir=None):
    df, result_dir = _load_benchmark_data(result_dir)

    fig, ax = plt.subplots(1, 1)
    grouped_times = df.groupby('build')['time']

    mean_times = grouped_times.mean()
    confidence_times = grouped_times.std() / np.sqrt(grouped_times.count())
    confidence_times *= 1.96

    mean_times.plot(kind='bar', yerr=confidence_times, ax=ax,
                    error_kw={'capsize': 10},
                    rot=0)

    for p in ax.patches:
        ax.annotate('{:.4f}'.format(p.get_height()),
                    (p.get_x() + p.get_width() / 4, p.get_height() / 2))

    ax.set_ylabel('Time (s)')
    ax.set_title(BASE_FILE_NAME)

    fig.tight_layout()

    plot_file_path = os.path.abspath(os.path.join(result_dir, 'result.png'))
    plt.savefig(plot_file_path)
    print('saved figure in {}'.format(plot_file_path))


run_matrix_vector_dotproduct_benchmarks()
plot_matrix_vector_dotproduct_benchmark()
