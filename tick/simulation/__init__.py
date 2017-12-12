# License: BSD 3 clause

from .features import features_normal_cov_uniform, \
    features_normal_cov_toeplitz
from .weights import weights_sparse_gauss, weights_sparse_exp

__all__ = [
    "weights_sparse_gauss",
    "weights_sparse_exp",
    "features_normal_cov_uniform",
    "features_normal_cov_toeplitz",
]
