# License: BSD 3 clause

import tick.base

from .weights import weights_sparse_gauss, weights_sparse_exp
from .features import features_normal_cov_uniform, features_normal_cov_toeplitz

__all__ = [
    "features_normal_cov_uniform", "features_normal_cov_toeplitz",
    "weights_sparse_exp", "weights_sparse_gauss"
]