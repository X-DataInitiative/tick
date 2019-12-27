import sys
import os

import numpy as np

import matplotlib.pyplot as plt

sys.path = ['/Users/martin/Projects/tick_jmlr/experiments'] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs



def plot_baseline(ax, baseline, transpose=True, remove_title=False):
    if transpose:
        baseline_matrix = np.array([baseline]).T
    else:
        baseline_matrix = np.array([baseline])

    ax.matshow(baseline_matrix, cmap='Blues', vmin=0)

    if not remove_title:
        ax.set_title('$\\mu_0$')

    if transpose:
        ax.set_xticks([])
    else:
        ax.set_yticks([])


def plot_adjacency(ax, adjacency, u, remove_y=False, remove_x=False,
                   remove_title=False, max_adjacency=None):
    ax.matshow(adjacency[:, :, u], cmap='Blues', vmin=0, vmax=max_adjacency)

    if not remove_title:
        ax.set_title('$(a_{{i, j, {}}})_{{0 \leq i, j \leq d}}$'.format(u))

    if remove_y:
        ax.set_yticks([])

    if remove_x:
        ax.set_xticks([])

    ax.xaxis.set_ticks_position('bottom')


def plot_coeffs_3_decays(coeffs_file_path, max_adjacency=None, ):
    n_decays = 3

    coeffs = np.load(coeffs_file_path, allow_pickle=True)
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

    return fig


def plot_coeffs_3_decays_5_coeffs(coeffs_file_paths, labels,
                                  max_adjacency=None):
    n_decays = 3

    fig_width, fig_height = 9, 12
    fig = plt.figure(figsize=(fig_width, fig_height))

    # hidden subplot to avoid auto call to tight_layout that is failing on Pycharm
    fig.add_subplot(10, 10, 19)

    n_rows = len(coeffs_file_paths)

    rects = []

    margin_left_baseline = 0.17
    margin_bottom = 0.02
    margin_top = 0.02
    margin_v_adjacency = 0.02
    margin_h_adjacency = 0.01

    canvas_height = 1 - margin_bottom - margin_top
    canvas_width = 1 - margin_left_baseline

    width_baseline = 0.06
    width_adjacency = (canvas_width - width_baseline) / \
                      n_decays - margin_h_adjacency

    height_adjacency = canvas_height / n_rows - margin_v_adjacency

    for i in range(n_rows):
        margin_left = margin_left_baseline

        bottom_adjacency = margin_bottom + \
                           canvas_height / n_rows * (n_rows - i - 1)

        height_baseline = height_adjacency
        bottom_baseline = bottom_adjacency

        rect_baseline = [margin_left, bottom_baseline,
                         width_baseline, height_baseline]

        margin_left += width_baseline

        rect_adjacencies = []
        for u in range(n_decays):
            if u > 0:
                margin_left += width_adjacency + margin_h_adjacency

            rect_adjacencies += [[margin_left, bottom_adjacency,
                                  width_adjacency, height_adjacency]]

        rects += [[rect_baseline] + rect_adjacencies]

        fig.text(margin_left_baseline / 2,
                 bottom_adjacency + height_adjacency * 0.7,
                 labels[i], fontsize=14, horizontalalignment='center')

    for coeffs_file_path, rect in zip(coeffs_file_paths, rects):
        print(coeffs_file_path)
        coeffs = np.load(coeffs_file_path, allow_pickle=True)
        baseline, adjacency = mus_alphas_from_coeffs(coeffs, n_decays)
        # dim = 3
        # baseline = np.random.rand(dim)
        # adjacency = np.random.rand(dim, dim, 3)

        remove_title = coeffs_file_paths.index(coeffs_file_path) != 0
        ax_baseline = fig.add_axes(rect[0])
        plot_baseline(ax_baseline, baseline, transpose=True,
                      remove_title=remove_title)

        for u in range(n_decays):
            ax_adjacency_u = fig.add_axes(rect[u + 1])
            max_adjacency = adjacency.max() if max_adjacency is None else max_adjacency

            remove_x = coeffs_file_paths.index(coeffs_file_path) != n_rows - 1


            plot_adjacency(ax_adjacency_u, adjacency, u,
                           remove_y=True, remove_x=remove_x,
                           remove_title=remove_title,
                           max_adjacency=max_adjacency)


    return fig


if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 3
    original_coeffs_file_path_ = os.path.join(
            os.path.dirname(__file__),
            'original_coeffs_dim_{}_decays_{}.npy'.format(dim_, n_decays_))
    originalcoeffs_ = np.load(original_coeffs_file_path_, allow_pickle=True)
    o_baseline_, o_adjacency_ = mus_alphas_from_coeffs(originalcoeffs_, n_decays_)

    coeffs_file_paths_ = [original_coeffs_file_path_]
    labels = ['Original', 'L1', 'wL1', 'L1Nuclear', 'wL1Nuclear']
    for prox_name in ['l1', 'l1w_un', 'l1_nuclear', 'l1w_un_nuclear']:
        coeffs_file_path_ = 'coeffs_{}_dim_{}_decays_{}_T_{}.npy'\
            .format(prox_name, dim_, n_decays_, 20000)
        coeffs_file_paths_ += [coeffs_file_path_]

    plot_coeffs_3_decays_5_coeffs(coeffs_file_paths_, labels,
                                  max_adjacency=o_adjacency_.max())

    plt.show()