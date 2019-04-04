# License: BSD 3 clause

import numpy as np
from numpy.linalg import svd

from tick.base_model import ModelGeneralizedLinear, ModelFirstOrder, \
    ModelLipschitz

from .build.linear_model import ModelLinRegDouble as _ModelLinRegDouble
from .build.linear_model import ModelLinRegFloat as _ModelLinRegFloat

__author__ = 'Stephane Gaiffas'


class ModelLinReg(ModelFirstOrder, ModelGeneralizedLinear, ModelLipschitz):
    """Least-squares loss for linear regression. This class gives first
    order information (gradient and loss) for this model and can be passed
    to any solver through the solver's ``set_model`` method.

    Given training data :math:`(x_i, y_i) \\in \\mathbb R^d \\times \\mathbb R`
    for :math:`i=1, \\ldots, n`, this model considers a goodness-of-fit

    .. math::
        f(w, b) = \\frac 1n \\sum_{i=1}^n \\ell(y_i, b + x_i^\\top w),

    where :math:`w \\in \\mathbb R^d` is a vector containing the model-weights,
    :math:`b \\in \\mathbb R` is the intercept (used only whenever
    ``fit_intercept=True``) and
    :math:`\\ell : \\mathbb R^2 \\rightarrow \\mathbb R` is the loss given by

    .. math::
        \\ell(y, y') = \\frac 12 (y - y')^2

    for :math:`y, y' \in \mathbb R`. Data is passed to this model through the
    ``fit(X, y)`` method where X is the features matrix (dense or sparse) and
    y is the vector of labels.

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

    _cpp_class_dtype_map = {
        np.dtype('float64'): _ModelLinRegDouble,
        np.dtype('float32'): _ModelLinRegFloat
    }

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
        output : `ModelLinReg`
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
        if self.dtype != coeffs.dtype:
            raise ValueError(
                "Model Linreg has received coeffs array with unexpected dtype")
        return self._model.loss(coeffs)

    def _get_lip_best(self):
        # TODO: Use sklearn.decomposition.TruncatedSVD instead?
        s = svd(self.features, full_matrices=False, compute_uv=False)[0] ** 2
        if self.fit_intercept:
            return (s + 1) / self.n_samples
        else:
            return s / self.n_samples

    def _build_cpp_model(self, dtype_or_object_with_dtype):
        model_class = self._get_typed_class(dtype_or_object_with_dtype,
                                            self._cpp_class_dtype_map)
        return model_class(self.features, self.labels, self.fit_intercept,
                           self.n_threads)

    def _get_params_set(self):
        """Get the set of parameters
        """
        return {
            *ModelFirstOrder._get_params_set(self),
            *ModelGeneralizedLinear._get_params_set(self),
            *ModelLipschitz._get_params_set(self),
            'n_threads'}

    @property
    def _AtomicClass(self):
        return AtomicModelLinReg


from .build.linear_model import ModelLinRegAtomicDouble as _ModelLinRegAtomicDouble
from .build.linear_model import ModelLinRegAtomicFloat as _ModelLinRegAtomicFloat


class AtomicModelLinReg(ModelLinReg):
    _cpp_class_dtype_map = {
        np.dtype('float32'): _ModelLinRegAtomicFloat,
        np.dtype('float64'): _ModelLinRegAtomicDouble
    }
