import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import rcParams


def plot_coeffs(mus, alphas, coeffs=None):
    dim = mus.shape[0]
    if len(alphas.shape) == 2:
        n_decays = 1
    else:
        _, _, n_decays = alphas.shape

    if coeffs is None:
        n_mu_plots = 1
        n_alpha_plots = n_decays
    elif n_decays == 1:
        n_mu_plots = 3
        n_alpha_plots = 3
    else:
        raise ValueError('Cannot plot error with sumexp')

    # Deal with mu
    rcParams['figure.figsize'] = 20, 1 * n_mu_plots
    ax_original = plt.subplot(n_mu_plots, 1, 1)
    ax_original.matshow([mus], cmap='Blues', vmin=0)
    ax_original.set_ylabel('mu org')

    if coeffs is not None:
        ax_estimated = plt.subplot(n_mu_plots, 1, 2)
        mus_ = coeffs[0:dim]
        ax_estimated.matshow([mus_], cmap='Blues', vmin=0)
        ax_estimated.set_ylabel('mu est')

        diff_matrix = mus_ - mus
        max_diff = np.max(np.abs(diff_matrix))
        ax_diff = plt.subplot(n_mu_plots, 1, 3)
        im_diff = ax_diff.matshow([diff_matrix], cmap='bwr', vmin=-max_diff,
                                  vmax=max_diff)
        ax_diff.set_ylabel('mu diff')

    plt.show()

    # Deal with alpha
    rcParams['figure.figsize'] = 4 * n_alpha_plots, 4
    for u in range(n_decays):
        matrix = alphas[:, :, u] if n_decays > 1 else alphas
        ax_original = plt.subplot(1, n_alpha_plots, u + 1)
        ax_original.matshow(matrix, cmap='Blues')
        ax_original.set_xlabel('original alphas {}'.format(u))

    if coeffs is not None:
        ax_estimated = plt.subplot(1, n_alpha_plots, 2)
        alphas_ = coeffs[dim:].reshape((dim, dim))
        ax_estimated.matshow(alphas_, cmap='Blues')
        ax_estimated.set_xlabel('estimated alphas')

        ax_diff = plt.subplot(1, n_alpha_plots, 3)
        diff_matrix = alphas_ - alphas
        max_diff = np.max(np.abs(diff_matrix))
        im_diff = ax_diff.matshow(diff_matrix, cmap='bwr', vmin=-max_diff,
                                  vmax=max_diff)
        ax_diff.set_xlabel('alphas diff')

        divider4 = make_axes_locatable(ax_diff)
        cax4 = divider4.append_axes("right", size="20%", pad=0.05)
        cbar4 = plt.colorbar(im_diff, cax=cax4)

    plt.show()


def plot_ticks_hist(ticks, title=''):
    n_ticks = [len(t) for t in ticks]
    rcParams['figure.figsize'] = 16, 3
    plt.step(np.arange(len(n_ticks)), n_ticks)
    plt.title(title)
    plt.xlabel('dimension')
    plt.ylabel('n jumps')
    plt.show()


if __name__ == '__main__':
    from experiments.hawkes_coeffs import coeffs_from_mus_alpha

    dim = 3
    mu = np.arange(dim)
    alpha = dim + np.arange(dim * dim).reshape(dim, dim)
    coeffs = coeffs_from_mus_alpha(mu, alpha) + np.random.rand(dim + dim * dim)
    plot_coeffs(mu, alpha, coeffs=coeffs)

