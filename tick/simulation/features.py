# License: BSD 3 clause

import numpy as np
from scipy.linalg.special_matrices import toeplitz


def features_normal_cov_uniform(n_samples: int = 200, n_features: int = 30,
                                dtype="float64"):
    """Normal features generator with uniform covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a specific covariance matrix given by 0.5 * (U + U.T),
    where U is uniform on [0, 1] and diagonal filled by ones.

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    C = np.random.uniform(size=(n_features, n_features), dtype=dtype)
    np.fill_diagonal(C, 1.0)
    cov = 0.5 * (C + C.T)
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def features_normal_cov_toeplitz(n_samples: int = 200, n_features: int = 30,
                                 cov_corr: float = 0.5, dtype="float64"):
    """Normal features generator with toeplitz covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a toeplitz covariance matrix

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    cov_corr : `float`, default=0.5
        correlation coefficient of the Toeplitz correlation matrix

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance

    """
    cov = toeplitz(cov_corr ** np.arange(0, n_features))
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features
