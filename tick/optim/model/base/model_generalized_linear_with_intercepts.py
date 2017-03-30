
from . import ModelLabelsFeatures

__author__ = 'Stephane Gaiffas'


class ModelGeneralizedLinearWithIntercepts(ModelLabelsFeatures):
    """An abstract base class for a generalized linear model (one-class
    supervised learning) with individual intercepts

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features) (read-only)
        The features matrix

    labels : `numpy.ndarray`, shape=(n_samples,)  (read-only)
        The labels vector

    n_samples : `int`  (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    def __init__(self):
        ModelLabelsFeatures.__init__(self)

    def _get_n_coeffs(self):
        return self.n_features + self.n_samples
