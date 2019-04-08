# License: BSD 3 clause

from warnings import warn
import numpy as np


def weights_sparse_gauss(n_weights: int = 100, nnz: int = 10, std: float = 1.,
                         dtype="float64") -> np.ndarray:
    """Sparse and gaussian model weights generator
    Instance of weights for a model, given by a sparse vector,
    where non-zero coordinates (chosen at random) are centered Gaussian
    with given standard-deviation

    Parameters
    ----------
    n_weights : `int`, default=100
        Number of weights

    nnz : `int`, default=10
        Number of non-zero weights

    std : `float`, default=1.
        Standard deviation of the Gaussian non-zero entries

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_weights,)
        The weights vector
    """
    if nnz >= n_weights:
        warn(("nnz must be smaller than n_weights "
              "using nnz=n_weights instead"))
        nnz = n_weights
    weights0 = np.zeros(n_weights, dtype=dtype)
    idx = np.arange(n_weights)
    np.random.shuffle(idx)
    weights0[idx[:nnz]] = np.random.randn(nnz)
    weights0 *= std
    return weights0


def weights_sparse_exp(n_weigths: int = 100, nnz: int = 10, scale: float = 10.,
                       dtype="float64") -> np.ndarray:
    """Sparse and exponential model weights generator

    Instance of weights for a model, given by a vector with
    exponentially decaying components: the j-th entry is given by

    .. math: (-1)^j \exp(-j / scale)

    for 0 <= j <= nnz - 1. For j >= nnz, the entry is zero.

    Parameters
    ----------
    n_weigths : `int`, default=100
        Number of weights

    nnz : `int`, default=10
        Number of non-zero weights

    scale : `float`, default=10.
        The scaling of the exponential decay

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : np.ndarray, shape=(n_weigths,)
        The weights vector
    """
    if nnz >= n_weigths:
        warn(("nnz must be smaller than n_weights "
              "using nnz=n_weigths instead"))
        nnz = n_weigths
    idx = np.arange(nnz)
    out = np.zeros(n_weigths, dtype=dtype)
    out[:nnz] = np.exp(-idx / scale)
    out[:nnz:2] *= -1
    return out
