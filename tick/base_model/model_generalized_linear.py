# License: BSD 3 clause

from . import ModelLabelsFeatures

__author__ = 'Stephane Gaiffas'


class ModelGeneralizedLinear(ModelLabelsFeatures):
    """An abstract base class for a generalized linear model (one-class
    supervised learning)

    Parameters
    ----------
    fit_intercept : `bool`
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

    dtype : `{'float64', 'float32'}`
        Type of the data arrays used.

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    _attrinfos = {
        "fit_intercept": {
            "writable": True,
            "cpp_setter": "set_fit_intercept"
        }
    }

    def __init__(self, fit_intercept: bool = True):
        ModelLabelsFeatures.__init__(self)
        self.fit_intercept = fit_intercept

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()
