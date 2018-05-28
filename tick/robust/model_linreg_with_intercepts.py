# License: BSD 3 clause

import numpy as np
from numpy.linalg import svd

from tick.base_model import ModelFirstOrder, ModelLipschitz
from .base import ModelGeneralizedLinearWithIntercepts
from .build.robust import \
    ModelLinRegWithInterceptsDouble as _ModelLinRegWithIntercepts

__author__ = 'Stephane Gaiffas'


class ModelLinRegWithIntercepts(
        ModelFirstOrder, ModelGeneralizedLinearWithIntercepts, ModelLipschitz):
    """Linear regression model with individual intercepts.
    This class gives first order information (gradient and loss) for this model

    Parameters
    ----------
    fit_intercept : `bool`, default=`True`
        If `True`, the model uses an intercept

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features) (read-only)
        The features matrix

    labels : `numpy.ndarray`, shape=(n_samples,) (read-only)
        The labels vector

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_threads : `int`, default=1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of threads available on
          the CPU
        * otherwise the desired number of threads
    """

    def __init__(self, fit_intercept: bool = True, n_threads: int = 1):
        ModelFirstOrder.__init__(self)
        ModelGeneralizedLinearWithIntercepts.__init__(self, fit_intercept)
        ModelLipschitz.__init__(self)
        self.n_threads = n_threads

    # TODO: implement _set_data and not fit
    def fit(self, features, labels):
        """Set the data into the model object

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : `ModelLinRegWithIntercepts`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels)
        ModelGeneralizedLinearWithIntercepts.fit(self, features, labels)
        ModelLipschitz.fit(self, features, labels)
        self._set(
            "_model",
            _ModelLinRegWithIntercepts(self.features, self.labels,
                                       self.fit_intercept, self.n_threads))
        return self

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_lip_best(self):
        s = svd(self.features, full_matrices=False, compute_uv=False)[0] ** 2
        if self.fit_intercept:
            return (s + 2) / self.n_samples
        else:
            return (s + 1) / self.n_samples
