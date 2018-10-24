import os

import numpy as np

from experiments.io_utils import get_coeffs_dir
from experiments.plot_hawkes import plot_coeffs
from tick.hawkes import SimuHawkesExpKernels


def dim_from_n(n):
    dim = int(round(0.5*(np.sqrt(1 + 4 * n) - 1)))
    return dim


def mus_alphas_from_coeffs(coeffs):
    n = coeffs.shape[0]
    dim = dim_from_n(n)
    mus = coeffs[0:dim]
    alphas = coeffs[dim:].reshape((dim, dim))
    return mus, alphas


def coeffs_from_mus_alpha(mus, alpha):
    return np.vstack((mus, alpha)).ravel()


def retrieve_coeffs(dim, directory_prefix):
    if dim == 30:
        betas, mu0, A0 = get_coeffs_dim_30()
    elif dim == 100:
        betas, mu0, A0 = get_coeffs_dim_100()
    else:
        raise ValueError('Unhandled number of nodes {}'.format(dim))

    check_existing_coeffs(dim, mu0, A0, directory_prefix)
    return betas, mu0, A0


def check_existing_coeffs(dim, mu0, A0, directory_prefix):
    original_coeffs = np.hstack((mu0, A0.reshape(dim * dim)))
    os.makedirs(get_coeffs_dir(dim, directory_prefix), exist_ok=True)
    original_coeffs_path = os.path.join(
        get_coeffs_dir(dim, directory_prefix), 'original_coeffs.npy')
    if os.path.exists(original_coeffs_path):
        previous_coeffs = np.load(original_coeffs_path)
        np.testing.assert_almost_equal(previous_coeffs, original_coeffs)
        print('coeffs file existed already and was the same')
    else:
        np.save(original_coeffs_path, original_coeffs)


def block_matrix(dimension: int = 100,
                 blocks_ranges: list = None,
                 min_value: float = 0., max_value: float = 0.2):
    A = np.zeros((dimension, dimension))
    for block_range in blocks_ranges:
        nodes = np.array(block_range, dtype=int)
        for a in nodes:
            for b in nodes:
                A[a, b] = min_value + (max_value - min_value) * \
                                      np.random.rand(1)
    A *= 0.8 / np.linalg.norm(A, 2)
    return A


def get_coeffs_dim_30(spectral_radius=0.8):
    dim = 30

    beta = 1
    betas = np.ones((dim, dim)) * beta

    blocks_ranges = [
        range(0, 7),
        range(4, 15),
        range(11, 20),
        range(20, 27),
        range(27, 29),
        range(29, 30),
    ]
    A0 = np.zeros((dim, dim))
    mu0 = np.zeros(dim)
    for i, block_range in enumerate(blocks_ranges):
        nodes = np.array(block_range, dtype=int)
        for a in nodes:
            for b in nodes:
                if i == 0:
                    A0[a, b] = 1
                elif i == 1:
                    A0[a, b] = 2
                elif i == 2:
                    A0[a, b] += 1
                elif i == 3:
                    A0[a, b] = 1
                elif i == 4:
                    A0[a, b] = 3
                elif i == 5:
                    A0[a, b] = 5
        a, b = block_range[0], block_range[-1] + 1
        if i == 0:
            mu0[a: b] = 0.3
        elif i == 1:
            mu0[a: b] = 0.1
        elif i == 2:
            mu0[a: b] += 0.2
        elif i == 3:
            mu0[a: b] = 0.1
        elif i == 4:
            mu0[a: b] = 0.8
        elif i == 5:
            mu0[a: b] = 1.0

    hawkes = SimuHawkesExpKernels(A0, betas, baseline=mu0)
    hawkes.adjust_spectral_radius(spectral_radius)

    return hawkes.decays, hawkes.baseline, hawkes.adjacency


def get_coeffs_dim_100(spectral_radius=0.8):
    dim = 100

    beta = 1
    betas = np.ones((dim, dim)) * beta

    blocks_ranges = [
        range(0, 12),
        range(10, 30),
        range(22, 60),
        range(60, 90),
        range(90, 97),
        range(97, 100)
    ]
    A0 = np.zeros((dim, dim))
    mu0 = np.zeros(dim)
    for i, block_range in enumerate(blocks_ranges):
        nodes = np.array(block_range, dtype=int)
        for a in nodes:
            for b in nodes:
                if i == 0:
                    A0[a, b] = 2
                elif i == 1:
                    A0[a, b] += 1
                elif i == 2:
                    A0[a, b] += 1.3
                elif i == 3:
                    A0[a, b] = 1.5
                elif i == 4:
                    A0[a, b] = 3
                elif i == 5:
                    A0[a, b] = 3
        a, b = block_range[0], block_range[-1] + 1
        if i == 0:
            mu0[a: b] = 0.3
        elif i == 1:
            mu0[a: b] = 0.1
        elif i == 2:
            mu0[a: b] += 0.2
        elif i == 3:
            mu0[a: b] = 0.1
        elif i == 4:
            mu0[a: b] = 0.8
        elif i == 5:
            mu0[a: b] = 1.0

    hawkes = SimuHawkesExpKernels(A0, betas, baseline=mu0)
    hawkes.adjust_spectral_radius(spectral_radius)

    return hawkes.decays, hawkes.baseline, hawkes.adjacency


if __name__ == '__main__':
    print(block_matrix(5, blocks_ranges=[range(0, 3), range(2, 5)]))

    betas, mu0, A0 = get_coeffs_dim_100()
    plot_coeffs(mu0, A0)

    dim = 30
    betas, mu0, A0 = get_coeffs_dim_30()
    plot_coeffs(mu0, A0)
    directory_prefix = '/Users/martin/Downloads/jmlr_hawkes_data/'

    retrieve_coeffs(dim, directory_prefix)

