# License: BSD 3 clause

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tick.prox import ProxL1

sys.path = [os.path.abspath('../../../')] + sys.path
sys.path = [os.path.abspath('../../')] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs, DECAYS_3
from tick.hawkes import SimuHawkesSumExpKernels

if __name__ == '__main__':
    dim_ = 10
    n_decays_ = 3
    end_time_ = 5000
    prox_ = ProxL1(0, positive=True)

    original_coeffs_file_path_ = os.path.join(
        os.path.dirname(__file__),
        'original_coeffs_dim_{}_decays_{}.npy'.format(dim_, n_decays_))
    originalcoeffs_ = np.load(original_coeffs_file_path_, allow_pickle=True)
    baseline, adjacency = mus_alphas_from_coeffs(originalcoeffs_, n_decays_)

    history_file_path_ = 'llh_vs_ls_convergence_speed_T_{}.pkl'.format(
        end_time_)
    simu = SimuHawkesSumExpKernels(
        adjacency, DECAYS_3, baseline=baseline, end_time=1000,
        verbose=False, seed=2111)

    simu.simulate()

    n_nodes = 30

    fig = plt.figure(figsize=(8, 4))
    ax_grid = gridspec.GridSpec(n_nodes, 1)
    ax_grid.update(hspace=0)  # set the spacing between axes.

    for i, timestamps in enumerate(simu.timestamps[:n_nodes]):
        ax = plt.subplot(ax_grid[i])
        t_min = 500
        t_max = 600
        selected_timestamps = timestamps[
            (timestamps >= t_min) & (timestamps <= t_max)]
        ax.vlines(selected_timestamps, ymin=0, ymax=1, lw=0.6)

        if i < n_nodes - 1:
            ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim([t_min, t_max])

    plt.show()
