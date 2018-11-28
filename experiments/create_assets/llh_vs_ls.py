import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.metrics_utils import get_confidence_interval_half_width

sys.path = [os.path.abspath('../../../')] + sys.path
sys.path = [os.path.abspath('../../')] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs, DECAYS_3
from tick.hawkes import (
    SimuHawkesSumExpKernels, ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik)


def run_simu_and_timings(adjacency, decays, baseline, end_time):
    simu = SimuHawkesSumExpKernels(
        adjacency, decays, baseline=baseline, end_time=end_time,
        verbose=False)

    simu.simulate()

    timestamps = simu.timestamps

    model_ls = ModelHawkesSumExpKernLeastSq(DECAYS_3)
    model_ls.fit(timestamps)

    model_llh = ModelHawkesSumExpKernLogLik(DECAYS_3)
    model_llh.fit(timestamps)

    start_weights_ls = time.clock()
    model_ls._model.compute_weights()
    time_weights_ls = time.clock() - start_weights_ls

    start_weights_llh = time.clock()
    model_llh._model.compute_weights()
    time_weights_llh = time.clock() - start_weights_llh

    start_loss_ls = time.clock()
    for _ in range(100):
        model_ls.loss(np.ones(model_ls.n_coeffs))
    time_loss_ls = time.clock() - start_loss_ls

    start_loss_llh = time.clock()
    for _ in range(100):
        model_llh.loss(np.ones(model_ls.n_coeffs))
    time_loss_llh = time.clock() - start_loss_llh

    return time_weights_ls, time_weights_llh, time_loss_ls, time_loss_llh


def generate_data(dim, n_decays, end_times, csv_path, n_tries):
    original_coeffs_file_path_ = os.path.join(
        os.path.dirname(__file__),
        'original_coeffs_dim_{}_decays_{}.npy'.format(dim, n_decays))
    originalcoeffs_ = np.load(original_coeffs_file_path_)
    baseline, adjacency = mus_alphas_from_coeffs(originalcoeffs_, n_decays)

    columns = ['dim', 'n_decays', 'end_time', 'n_tries',
               'time_weights_ls', 'time_weights_ls_std',
               'time_weights_llh', 'time_weights_llh_std',
               'time_loss_ls', 'time_loss_ls_std',
               'time_loss_llh', 'time_loss_llh_std']
    df = pd.DataFrame(columns=columns)
    for end_time in end_times:
        print('Generating: {}'.format(end_time))

        sample = []
        for _ in range(n_tries):
            time_weights_ls, time_weights_llh, time_loss_ls, time_loss_llh = \
                run_simu_and_timings(adjacency, DECAYS_3, baseline, end_time)

            sample += [[time_weights_ls, time_weights_llh, time_loss_ls,
                       time_loss_llh]]

        sample = np.array(sample)
        row = [dim, n_decays, end_time, n_tries]
        for i in range(sample.shape[1]):
            row += [sample[:, i].mean(), sample[:, i].std()]

        df = pd.concat([df, pd.DataFrame([row], columns=columns)],
                       ignore_index=True)

    df.to_csv(csv_path, index=False)


def plot_with_ci(x, y, y_std, n_tries, label, ax, color):
    hw_cis = get_confidence_interval_half_width(
        y_std, n_tries, confidence=0.95)

    ax.fill_between(x, y + hw_cis, y - hw_cis, alpha=0.3, color=color)
    ax.plot(x, y, color=color, label=label)


if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 3
    end_times_ = [5000, 7000, 10000, 15000, 20000, 25000, 30000]
    n_tries_ = 5

    csv_path_ = 'timings_llh_vs_ls_100_true.csv'
    # generate_data(dim_, n_decays_, end_times_, csv_path_, n_tries_)

    df_timings = pd.read_csv(csv_path_)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    df_timings = df_timings[(df_timings['dim'] == dim_)
                            & (df_timings['n_decays'] == n_decays_)]

    ax_weigths = axes[0]
    plot_with_ci(df_timings['end_time'], df_timings['time_weights_llh'],
                 df_timings['time_weights_llh_std'], df_timings['n_tries'],
                 'log-likelihood', ax_weigths, 'C0')
    plot_with_ci(df_timings['end_time'], df_timings['time_weights_ls'],
                 df_timings['time_weights_ls_std'], df_timings['n_tries'],
                 'least squares', ax_weigths, 'C1')
    ax_weigths.set_title('Weights computation', fontsize=16)
    ax_weigths.set_ylabel('time (s)', fontsize=13)

    ax_loss = axes[1]
    plot_with_ci(df_timings['end_time'], df_timings['time_loss_llh'],
                 df_timings['time_loss_llh_std'], df_timings['n_tries'],
                 'log-likelihood', ax_loss, 'C0')
    plot_with_ci(df_timings['end_time'], df_timings['time_loss_ls'],
                 df_timings['time_loss_ls_std'], df_timings['n_tries'],
                 'least squares', ax_loss, 'C1')
    ax_loss.set_yscale('log')
    ax_loss.set_title('100 loss computations', fontsize=16)
    ax_loss.set_ylabel('time in log scale (s)', fontsize=13)

    ax_loss.legend()
    plt.show()
