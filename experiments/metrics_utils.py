# License: BSD 3 clause

import numpy as np
import scipy.stats as sps


def mean_and_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std


def get_confidence_interval_half_width(std, n_samples, confidence=0.95):
    score = sps.norm.ppf(1 - (1 - confidence) / 2)  # usually 1.96
    return score * std / np.sqrt(n_samples)


def strength_range_from_infos(infos):
    strength_range = list(infos[0]['alpha_auc'].keys())
    # 2d case
    if isinstance(strength_range[0], tuple):
        pass
    else:
        strength_range = np.array(strength_range)
    strength_range.sort()
    return strength_range


def extract_metric(metric, infos):
    """Returns for on metric an array containing the value of the metric for
    each penalty strength (in column) and each model (in row)
    """
    strength_range = strength_range_from_infos(infos)
    return np.array([
        np.array([infos[i][metric].get(s, np.nan) for s in strength_range])
        for i in range(len(infos))
    ])


if __name__ == '__main__':
    sample_metric = np.array([
        [0.2, 0.2, 1.],
        [0.4, 0.35, 1.],
        [0.3, 0.3, 1.],
        [0.5, np.nan, 1.]
    ])
    print(sample_metric)
    means, std = mean_and_std(sample_metric)
    print(means)
    print(std)
    print(get_confidence_interval_half_width(std, sample_metric.shape[0]))
    print('----')

    infos = {
        0: {
            'alpha_auc': {
                0.0: 0.64068, 1.0e-05: 0.64718, 0.00031: 0.61906,
                0.01: 0.36259},
            'estimation_error': {
                0.0: 0.64061, 1.0e-05: 0.647, 0.00031: 0.61907, 0.01: 0.36257},
            'kendall': {
                0.0: 7.83e-05, 1.0e-05: -0.00105, 0.00031: 0.195, 0.01: 0.026},
            'kendall_no_diag': {
                0.0: 1.06805e-05, 1.0e-05: 7.13587e-05, 0.00031: -7.49639e-05,
                0.01: -0.00106}
        },
        1: {
            'alpha_auc': {
                0.0: 0.64063, 1.0e-05: 0.64706, 0.00031: 0.61907,
                0.01: 0.36263},
            'estimation_error': {
                0.0: 0.64064, 1.0e-05: 0.64717, 0.00031: 0.61903,
                0.01: 0.36262},
            'kendall': {
                0.0: -7.42597e-05, 1.0e-05: -0.00098, 0.00031: 0.19586,
                0.01: 0.02690},
            'kendall_no_diag': {
                0.0: -7.22135e-05, 1.0e-05: -1.66231e-05,
                0.00031: -6.06590e-05, 0.01: -0.00104}
        },
        2: {
            'alpha_auc': {
                0.0: 0.64073, 1.0e-05: 0.64702, 0.00031: 0.61902,
                0.01: 0.36257},
            'estimation_error': {
                0.0: 0.64061, 1.0e-05: 0.64702, 0.00031: 0.61912,
                0.01: 0.36258},
            'kendall': {
                0.0: -8.55895e-05, 1.0e-05: -0.00112, 0.00031: 0.19578,
                0.01: 0.02695},
            'kendall_no_diag': {
                0.0: 4.18291e-06, 1.0e-05: 5.15576e-05, 0.00031: -6.25331e-05,
                0.01: -0.00096}
        }
    }

    print('strength_range_from_infos', strength_range_from_infos(infos))
    print('extract_metric', extract_metric('alpha_auc', infos))
