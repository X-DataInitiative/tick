# License: BSD 3 clause

import numpy as np
from numpy.linalg import svd

from tick.base_model import ModelGeneralizedLinear, ModelFirstOrder, \
    ModelLipschitz
from .build.linear_model import ModelLogRegDouble as _ModelLogRegDouble
from .build.linear_model import ModelLogRegFloat as _ModelLogRegFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype('float32'): _ModelLogRegFloat,
    np.dtype('float64'): _ModelLogRegDouble
}


class ModelLogReg(ModelFirstOrder, ModelGeneralizedLinear, ModelLipschitz):
    """Logistic regression model for binary classification. This class gives
    first order information (gradient and loss) for this model and can be passed
    to any solver through the solver's ``set_model`` method.

    Given training data :math:`(x_i, y_i) \\in \\mathbb R^d \\times \\{ -1, 1 \\}`
    for :math:`i=1, \\ldots, n`, this model considers a goodness-of-fit

    .. math::
        f(w, b) = \\frac 1n \\sum_{i=1}^n \\ell(y_i, b + x_i^\\top w),

    where :math:`w \\in \\mathbb R^d` is a vector containing the model-weights,
    :math:`b \\in \\mathbb R` is the intercept (used only whenever
    ``fit_intercept=True``) and
    :math:`\\ell : \\mathbb R^2 \\rightarrow \\mathbb R` is the loss given by

    .. math::
        \\ell(y, y') = \\log(1 + \\exp(-y y'))

    for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`. Data is passed
    to this model through the ``fit(X, y)`` method where X is the features
    matrix (dense or sparse) and y is the vector of labels.

    Parameters
    ----------
    fit_intercept : `bool`
        If `True`, the model uses an intercept

    Attributes
    ----------
    features : {`numpy.ndarray`, `scipy.sparse.csr_matrix`}, shape=(n_samples, n_features)
        The features matrix, either dense or sparse

    labels : `numpy.ndarray`, shape=(n_samples,) (read-only)
        The labels vector

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    dtype : `{'float64', 'float32'}`
        Type of the data arrays used.

    n_threads : `int`, default=1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of threads available on
          the CPU
        * otherwise the desired number of threads
    """

    def __init__(self, fit_intercept: bool = True, n_threads: int = 1):
        ModelFirstOrder.__init__(self)
        ModelGeneralizedLinear.__init__(self, fit_intercept)
        ModelLipschitz.__init__(self)
        self.n_threads = n_threads

    # TODO: implement _set_data and not fit
    def fit(self, features, labels):
        """Set the data into the model object

        Parameters
        ----------
        features : {`numpy.ndarray`, `scipy.sparse.csr_matrix`}, shape=(n_samples, n_features)
            The features matrix, either dense or sparse

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : `ModelLogReg`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels)
        ModelGeneralizedLinear.fit(self, features, labels)
        ModelLipschitz.fit(self, features, labels)

        self._set("_model", self._build_cpp_model(features.dtype))
        return self

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    @staticmethod
    def sigmoid(coeffs: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """The sigmoid function

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Input vector

        out : `numpy.ndarray`
            Output vector

        Returns
        -------
        output : `numpy.ndarray`
            The sigmoid applied element-wise on ``coeffs``. Same as
            ``out``
        """
        if out is None:
            out = np.empty(coeffs.shape[0], dtype=coeffs.dtype)
        # sigmoid is a templated static function so
        ## we must call the right version for the right dtype
        dtype_map[coeffs.dtype].sigmoid(coeffs, out)
        return out

    def _get_lip_best(self):
        # TODO: Use sklearn.decomposition.TruncatedSVD instead?
        s = svd(self.features, full_matrices=False, compute_uv=False)[0] ** 2
        if self.fit_intercept:
            return (s + 1) / (4 * self.n_samples)
        else:
            return s / (4 * self.n_samples)

    def _build_cpp_model(self, dtype_or_object_with_dtype):
        model_class = self._get_typed_class(dtype_or_object_with_dtype,
                                            dtype_map)
        return model_class(self.features, self.labels, self.fit_intercept,
                           self.n_threads)
