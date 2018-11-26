import sys
import os

import numpy as np

import matplotlib.pyplot as plt

sys.path = ['/Users/martin/Projects/tick_jmlr/experiments'] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs


def plot_baseline(ax, baseline, transpose=True):
    if transpose:
        baseline_matrix = np.array([baseline]).T
    else:
        baseline_matrix = np.array([baseline])

    ax.matshow(baseline_matrix, cmap='Blues', vmin=0)
    ax.set_title('$\\mu_0$')

    if transpose:
        ax.set_xticks([])
    else:
        ax.set_yticks([])


def plot_adjacency(ax, adjacency, u, remove_y=False, max_adjacency=None):
    ax.matshow(adjacency[:, :, u], cmap='Blues', vmin=0, vmax=max_adjacency)
    ax.set_title('$(a_{{i, j, {}}})_{{0 \leq i, j \leq d}}$'.format(u))

    if remove_y:
        ax.set_yticks([])

    ax.xaxis.set_ticks_position('bottom')


def plot_coeffs_3_decays(coeffs_file_path, max_adjacency=None):
    n_decays = 3

    coeffs = np.load(coeffs_file_path)
    baseline, adjacency = mus_alphas_from_coeffs(coeffs, n_decays)

    fig_width, fig_height = 8.2, 3.3
    fig = plt.figure(figsize=(fig_width, fig_height))

    # hidden subplot to avoid auto call to tight_layout that is failing on Pycharm
    fig.add_subplot(10, 10, 9 * 8 + 1)

    margin_baseline = 0.
    width_baseline = 0.09
    margin_left = margin_baseline

    bottom_adjacency = 0.07
    height_adjacency = 0.8
    width_adjacency = 0.29
    margin_h_adjacency = 0.01

    height_baseline = min(height_adjacency,
                          width_adjacency * fig_width / fig_height)
    bottom_baseline = bottom_adjacency + (height_adjacency - height_baseline) / 2

    rect_baseline = [margin_left, bottom_baseline,
                     width_baseline, height_baseline]

    margin_left += width_baseline + margin_h_adjacency

    rect_adjacencies = []
    for u in range(n_decays):
        if u > 0:
            margin_left += width_adjacency + margin_h_adjacency

        rect_adjacencies += [[margin_left, bottom_adjacency,
                              width_adjacency, height_adjacency]]

    ax_baseline = fig.add_axes(rect_baseline)
    plot_baseline(ax_baseline, baseline, transpose=True)

    for u in range(n_decays):
        ax_adjacency_u = fig.add_axes(rect_adjacencies[u])
        max_adjacency = adjacency.max() if max_adjacency is None else max_adjacency
        plot_adjacency(ax_adjacency_u, adjacency, u,
                       remove_y=True, max_adjacency=max_adjacency)

    plt.show()


if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 3
    original_coeffs_file_path_ = os.path.join(
            os.path.dirname(__file__),
            'original_coeffs_dim_{}_decays_{}.npy'.format(dim_, n_decays_))
    originalcoeffs_ = np.load(original_coeffs_file_path_)
    o_baseline_, o_adjacency_ = mus_alphas_from_coeffs(originalcoeffs_, n_decays_)

    coeffs_file_path_ = 'coeffs_l1_dim_{}_decays_{}_T_{}.npy'\
        .format(dim_, n_decays_, 20000)
    plot_coeffs_3_decays(coeffs_file_path_, max_adjacency=o_adjacency_.max())