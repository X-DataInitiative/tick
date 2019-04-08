# License: BSD 3 clause

from tick.base_model import ModelGeneralizedLinear

__author__ = 'Stephane Gaiffas'


class ModelGeneralizedLinearWithIntercepts(ModelGeneralizedLinear):
    """An abstract base class for a generalized linear model (one-class
    supervised learning) with individual intercepts

    Parameters
    ----------
    fit_intercept : `bool`, default=`True`
        If `True`, the model uses an intercept

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

    def __init__(self, fit_intercept: bool = True):
        ModelGeneralizedLinear.__init__(self, fit_intercept)
