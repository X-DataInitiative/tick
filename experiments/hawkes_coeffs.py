import os

import numpy as np

from experiments.io_utils import get_coeffs_dir
from experiments.plot_hawkes import plot_coeffs

try:
    from tick.hawkes import SimuHawkesExpKernels, SimuHawkesSumExpKernels
except ImportError:
    print('tick not correctly installed')

DECAY_1 = 1.
DECAYS_3 = np.array([0.5, 2., 5.])


def dim_from_n(n, n_decays):
    dim = int(round(0.5*(np.sqrt(1 + 4 * n / n_decays) - 1)))
    return dim


def mus_alphas_from_coeffs(coeffs, n_decays):
    n = coeffs.shape[0]
    dim = dim_from_n(n, n_decays)
    mus = coeffs[0:dim]
    if n_decays == 1:
        alphas = coeffs[dim:].reshape((dim, dim))
    else:
        alphas = coeffs[dim:].reshape((dim, dim, n_decays))
    return mus, alphas


def coeffs_from_mus_alpha(mus, alpha):
    return np.hstack((mus, alpha.ravel())).ravel()


def retrieve_coeffs(dim, n_decays, directory_prefix):
    if dim == 10:
        betas, mu0, A0 = get_coeffs_dim_10(n_decays)
    elif dim == 30:
        # Dirty way to infer version for directory prefix
        # Useful to avoid to propagate version in all functions...
        if 'v2' in directory_prefix:
            version = 2
        elif 'v3' in directory_prefix:
            version = 3
        else:
            version = 1
        betas, mu0, A0 = get_coeffs_dim_30(n_decays, version)
    elif dim == 100:
        betas, mu0, A0 = get_coeffs_dim_100(n_decays)
    else:
        raise ValueError('Unhandled number of nodes {}'.format(dim))

    check_existing_coeffs(dim, mu0, A0, n_decays, directory_prefix)
    return betas, mu0, A0


def check_existing_coeffs(dim, mu0, A0, n_decays, directory_prefix):
    original_coeffs = np.hstack((mu0, A0.reshape(dim * dim * n_decays)))
    os.makedirs(get_coeffs_dir(dim, n_decays, directory_prefix), exist_ok=True)
    original_coeffs_path = os.path.join(
        get_coeffs_dir(dim, n_decays, directory_prefix),
        'original_coeffs.npy'.format(n_decays))
    if os.path.exists(original_coeffs_path):
        previous_coeffs = np.load(original_coeffs_path, allow_pickle=True)
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


def baseline_matrix_from_block_range(blocks_ranges, decays, spectral_radius, noise_level=0., seed=23983):
    dim = blocks_ranges[-1][0].stop
    n_decays = 1 if len(decays.shape) == 2 else decays.shape[0]
    r = np.random.RandomState(seed)  # for dim 10

    if n_decays == 1:
        A0 = np.zeros((dim, dim))
        mu0 = np.zeros(dim)
        for i, (block_range, coeff_mu, coeff_A) in enumerate(blocks_ranges):
            nodes = np.array(block_range, dtype=int)
            for a in nodes:
                for b in nodes:
                    A0[a, b] += coeff_A
            a, b = block_range[0], block_range[-1] + 1
            mu0[a: b] += coeff_mu
    else:
        decay_coeffs = r.rand(n_decays, len(blocks_ranges))

        A0 = np.zeros((dim, dim, n_decays))
        mu0 = np.zeros(dim)
        for i, (block_range, coeff_mu, coeff_A) in enumerate(blocks_ranges):
            nodes = np.array(block_range, dtype=int)
            for a in nodes:
                for b in nodes:
                    for u in range(n_decays):
                        A0[a, b, u] += coeff_A * decay_coeffs[u, i]
            a, b = block_range[0], block_range[-1] + 1
            mu0[a: b] += coeff_mu

    noise_A0 = r.rand(*A0.shape)
    noisy_A0 = (1 - noise_level) * A0 + noise_level * noise_A0

    noise_mu0 = r.rand(*mu0.shape)
    noisy_mu0 = (1 - noise_level) * mu0 + noise_level * noise_mu0

    simu_class = SimuHawkesExpKernels if n_decays == 1 \
        else SimuHawkesSumExpKernels
    hawkes = simu_class(noisy_A0, decays, baseline=noisy_mu0)
    hawkes.adjust_spectral_radius(spectral_radius)

    return hawkes.baseline, hawkes.adjacency


def get_coeffs_dim_10(n_decays, spectral_radius=0.8):
    dim = 10

    if n_decays == 1:
        beta = DECAY_1
        decays = np.ones((dim, dim)) * beta
    else:
        decays = DECAYS_3

    blocks_ranges = [
        (range(0, 4), 0.3, 1.3),
        (range(2, 7), 0.1, 1.),
        (range(7, 10), 0.2, 2),
    ]

    baseline, adjacency = baseline_matrix_from_block_range(
        blocks_ranges, decays, spectral_radius
    )

    return decays, baseline, adjacency


def get_coeffs_dim_30(n_decays, version, spectral_radius=0.8):
    dim = 30

    if n_decays == 1:
        beta = DECAY_1
        decays = np.ones((dim, dim)) * beta
    else:
        decays = DECAYS_3

    if version == 1:
        blocks_ranges = [
            (range(0, 7), 0.3, 1),
            (range(4, 15), 0.1, 2),
            (range(11, 20), 0.2, 1),
            (range(20, 27), 0.1, 1),
            (range(27, 29), 0.8, 3),
            (range(29, 30), 1.0, 5),
        ]

        baseline, adjacency = baseline_matrix_from_block_range(
            blocks_ranges, decays, spectral_radius
        )
    elif version == 2:
        blocks_ranges = [
            (range(0, 2), 0.3, 0.2),
            (range(2, 6), 0.4, 0.4),
            (range(6, 15), 0.1, 0.3),
            (range(15, 30), 0.2, 0.2),
        ]

        baseline, adjacency = baseline_matrix_from_block_range(
            blocks_ranges, DECAYS_3, 0.8, noise_level=0.03, seed=109230
        )
    elif version == 3:
        blocks_ranges = [
            (range(0, 10), 0.3, 0.2),
            (range(5, 15), 0.4, 0.4),
            (range(8, 22), 0.1, 0.3),
            (range(15, 25), 0.2, 0.2),
            (range(20, 30), 0.2, 0.2),
        ]

        baseline, adjacency = baseline_matrix_from_block_range(
            blocks_ranges, DECAYS_3, 0.8, noise_level=0.08, seed=109230
        )
    else:
        raise NotImplementedError('Unknown coeffs version')

    return decays, baseline, adjacency


def get_coeffs_dim_100(n_decays, spectral_radius=0.8):
    dim = 100

    if n_decays == 1:
        beta = DECAY_1
        decays = np.ones((dim, dim)) * beta
    else:
        decays = DECAYS_3

    blocks_ranges = [
        (range(0, 12), 0.3, 2),
        (range(10, 30), 0.1, 1),
        (range(22, 60), 0.2, 1.3),
        (range(60, 90), 0.1, 1.5),
        (range(90, 97), 0.8, 3),
        (range(97, 100), 1., 3),
    ]

    baseline, adjacency = baseline_matrix_from_block_range(
        blocks_ranges, decays, spectral_radius
    )

    return decays, baseline, adjacency


if __name__ == '__main__':
    import itertools

    print(block_matrix(5, blocks_ranges=[range(0, 3), range(2, 5)]))

    betas_, mu0_, A0_ = get_coeffs_dim_100(3)
    plot_coeffs(mu0_, A0_)

    dim_ = 30
    betas, mu0, A0 = get_coeffs_dim_30(3, spectral_radius=0.8)
    plot_coeffs(mu0, A0)
    directory_prefix_ = '/Users/martin/Downloads/jmlr_hawkes_data/'
    
    for dim_, n_decays_ in itertools.product([30, 100], [1, 3]):
        print('retrieved n_nodes={}, n_decays={}'.format(dim_, n_decays_))
        retrieve_coeffs(dim_, n_decays_, directory_prefix_)
