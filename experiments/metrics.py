from collections import OrderedDict

import numpy as np
import scipy.stats as sps
from sklearn.metrics import roc_auc_score

from experiments.hawkes_coeffs import mus_alphas_from_coeffs


def estimation_error(original_coeffs, coeffs, n_decays, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs, n_decays)
    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy(), n_decays)
    n_nodes = original_alphas.shape[0]

    original = original_alphas.ravel()
    estimated = alphas_.ravel()

    if remove_diag:
        original = remove_diag_flat_array(original, n_nodes, n_decays)
        estimated = remove_diag_flat_array(estimated, n_nodes, n_decays)

    return np.linalg.norm(original - estimated) ** 2 / np.linalg.norm(original)


def alphas_auc(original_coeffs, coeffs, n_decays, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs, n_decays)
    n_nodes = original_alphas.shape[0]

    non_zeros = original_alphas != 0
    labels = np.zeros_like(original_alphas)
    labels[non_zeros] = 1.

    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy(), n_decays)
    alphas_ = (alphas_ - np.min(alphas_)) * (np.max(alphas_) - np.min(alphas_))

    # let's flatten this
    labels = labels.ravel()
    probas = alphas_.ravel()

    if remove_diag:
        labels = remove_diag_flat_array(labels, n_nodes, n_decays)
        probas = remove_diag_flat_array(probas, n_nodes, n_decays)

    return roc_auc_score(labels, probas)


def kendall(original_coeffs, coeffs, n_decays, remove_diag=False):
    _, original_alphas = mus_alphas_from_coeffs(original_coeffs, n_decays)
    _, alphas_ = mus_alphas_from_coeffs(coeffs.copy(), n_decays)
    n_nodes = original_alphas.shape[0]

    original = original_alphas.ravel()
    estimated = alphas_.ravel()

    if remove_diag:
        original = remove_diag_flat_array(original, n_nodes, n_decays)
        estimated = remove_diag_flat_array(estimated, n_nodes, n_decays)

    return sps.kendalltau(original, estimated)[0]


def remove_diag_flat_array(arr, n_nodes, n_decays):
    arr = arr.copy()
    unused_value = -103901932
    for u in range(n_decays):
        arr[u::(n_nodes * n_decays) + n_decays] = unused_value
    return arr[arr != unused_value]


def get_metrics(original_coeffs=None, n_decays=None):
    """Having original_coeffs is useulf when we only want to have the
    informations on the metrics
    """
    metrics = OrderedDict()

    metrics["alpha_auc"] = {
        'evaluator': lambda x: alphas_auc(original_coeffs, x, n_decays),
        'best': 'max'
    }
    metrics["alphas_auc_no_diag"] = {
        'evaluator': lambda x: alphas_auc(original_coeffs, x, n_decays,
                                          remove_diag=True),
        'best': 'max'
    }
    metrics["estimation_error"] = {
        'evaluator': lambda x: estimation_error(original_coeffs, x, n_decays),
        'best': 'min'
    }
    metrics["estimation_error_no_diag"] = {
        'evaluator': lambda x: estimation_error(original_coeffs, x, n_decays,
                                                remove_diag=True),
        'best': 'min'
    }
    metrics["kendall"] = {
        'evaluator': lambda x: kendall(original_coeffs, x, n_decays),
        'best': 'max'
    }
    metrics["kendall_no_diag"] = {
        'evaluator': lambda x: kendall(original_coeffs, x, n_decays,
                                       remove_diag=True),
        'best': 'max'
    }

    return metrics


def compute_metrics(original_coeffs, coeffs, n_decays, info, strength):
    metrics = get_metrics(original_coeffs, n_decays)

    for metric, metric_info in metrics.items():
        evaluator = metric_info['evaluator']
        if metric not in info:
            info[metric] = {}
        info[metric][strength] = evaluator(coeffs)


if __name__ == '__main__':
    n_nodes_ = 4
    n_decays_ = 3

    coeffs_ = np.arange(n_nodes_ * n_nodes_ * n_decays_)
    if n_decays_ == 1:
        coeffs_ = coeffs_.reshape(n_nodes_, n_nodes_)
    else:
        coeffs_ = coeffs_.reshape(n_nodes_, n_nodes_, n_decays_)

    if n_decays_ == 1:
        print(coeffs_)
    else:
        for u in range(n_decays_):
            print(coeffs_[:, :, u])
    print(remove_diag_flat_array(coeffs_.ravel(), n_nodes_, n_decays_))
