import numpy as np
import scipy.stats as sps
from sklearn.metrics import roc_auc_score

from experiments.hawkes_coeffs import mus_alphas_from_coeffs


def estimation_error(original_coeffs, coeffs, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs)
    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy())
    dim = original_alphas.shape[0]

    original = original_alphas.reshape(dim * dim)
    estimated = alphas_.reshape(dim * dim)

    if remove_diag:
        original = remove_diag_flat_array(original, dim)
        estimated = remove_diag_flat_array(estimated, dim)

    return np.linalg.norm(original - estimated) ** 2 / np.linalg.norm(original)


def alphas_auc(original_coeffs, coeffs, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs)
    dim = original_alphas.shape[0]

    non_zeros = original_alphas != 0
    labels = np.zeros_like(original_alphas)
    labels[non_zeros] = 1.

    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy())
    alphas_ = (alphas_ - np.min(alphas_)) * (np.max(alphas_) - np.min(alphas_))

    # let's flatten this
    labels = labels.reshape(dim * dim)
    probas = alphas_.reshape(dim * dim)

    if remove_diag:
        labels = remove_diag_flat_array(labels, dim)
        probas = remove_diag_flat_array(probas, dim)

    return roc_auc_score(labels, probas)


def kendall(original_coeffs, coeffs, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs)
    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy())
    dim = original_alphas.shape[0]

    original = original_alphas.reshape(dim * dim)
    estimated = alphas_.reshape(dim * dim)

    if remove_diag:
        original = remove_diag_flat_array(original, dim)
        estimated = remove_diag_flat_array(estimated, dim)

    return sps.kendalltau(original, estimated)[0]


def remove_diag_flat_array(arr, dim):
    arr = arr.copy()
    unused_value = -103901932
    arr[::dim + 1] = unused_value
    return arr[arr != unused_value]
