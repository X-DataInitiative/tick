# License: BSD 3 clause

import numpy as np
from tick.base_model import ModelFirstOrder, ModelLipschitz
from .build.survival import ModelSCCS as _ModelSCCS
from tick.preprocessing.utils import check_longitudinal_features_consistency, \
    check_censoring_consistency


class ModelSCCS(ModelFirstOrder, ModelLipschitz):
    """Discrete-time Self Control Case Series (SCCS) likelihood. This class
    provides first order information (gradient and loss) model.

    Parameters
    ----------
    n_intervals : `int`
        Number of time intervals observed for each sample.

    n_lags : `numpy.ndarray`, shape=(n_features,), dtype="uint64"
        Number of lags per feature. The model will regress labels on the last
        observed values of the features over the corresponding `n_lags` time
        intervals. `n_lags` values must be between 0 and `n_intervals` - 1.

    Attributes
    ----------
    features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
        list of length n_cases, each element of the list of
        shape=(n_intervals, n_features)
        The list of features matrices.

    labels : `list` of `numpy.ndarray`,
        list of length n_cases, each element of the list of
        shape=(n_intervals,)
        The labels vector

    censoring : `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
        The censoring data. This array should contain integers in
        [1, n_intervals]. If the value i is equal to n_intervals, then there
        is no censoring for sample i. If censoring = c < n_intervals, then
        the observation of sample i is stopped at interval c, that is, the
        row c - 1 of the corresponding matrix. The last n_intervals - c rows
        are then set to 0.

    n_cases : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model
    """

    _const_attr = [
        "labels", "features", "censoring", "n_features", "n_cases", "n_lags",
        "n_intervals"
    ]

    _attrinfos = {key: {'writable': False} for key in _const_attr}

    def __init__(self, n_intervals: int, n_lags: np.array):
        ModelFirstOrder.__init__(self)
        ModelLipschitz.__init__(self)
        self.n_intervals = n_intervals
        self.n_features = len(n_lags)
        self.n_lags = n_lags
        for n_l in n_lags:
            if n_l >= n_intervals:
                raise ValueError("n_lags should be < n_intervals")
        self.labels = None
        self.features = None
        self.censoring = None
        self.n_cases = None

    def fit(self, features, labels, censoring=None):
        """Set the data into the model object.

        Parameters
        ----------
        features : List[{2d array, csr matrix containing float64
            of shape (n_intervals, n_features)}]
            The features matrix

        labels : List[{1d array, csr matrix of shape (n_intervals,)]
            The labels vector

        censoring : 1d array of shape (n_cases,)
            The censoring vector

        Returns
        -------
        output : `ModelSCCS`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels, censoring)
        ModelLipschitz.fit(self, features, labels)

        self._set(
            "_model",
            _ModelSCCS(self.features, self.labels, self.censoring, self.n_lags))

        self.dtype = features[0].dtype
        return self

    def _set_data(self, features, labels, censoring):
        """Set the data to the model.

        Parameters
        ----------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : `list` of `numpy.ndarray`,
            list of length n_cases, each element of the list of
            shape=(n_intervals,)
            The labels vector

        censoring : `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the correponding matrix. The last n_intervals - c rows
            are then set to 0.
        """
        n_intervals, n_coeffs = features[0].shape
        n_lags = self.n_lags
        self._set("n_intervals", n_intervals)
        self._set("n_coeffs", n_coeffs)
        # TODO: implement checker as outside function
        # if n_lags > 0 and n_coeffs % (n_lags + 1) != 0:
        #     raise ValueError("(n_lags + 1) should be a divisor of n_coeffs")
        # else:
        #     self._set("n_features", int(n_coeffs / (n_lags + 1)))
        self._set("n_cases", len(features))
        if len(labels) != self.n_cases:
            raise ValueError("Features and labels lists should have the same\
             length.")
        if censoring is None:
            censoring = np.full(self.n_cases, self.n_intervals, dtype="uint64")
        censoring = check_censoring_consistency(censoring, self.n_cases)
        features = check_longitudinal_features_consistency(
            features, (n_intervals, n_coeffs), "float64")
        labels = check_longitudinal_features_consistency(
            labels, (self.n_intervals,), "int32")

        self._set("labels", labels)
        self._set("features", features)
        self._set("censoring", censoring)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()

    def _get_lip_best(self):
        raise NotImplementedError("ModelSCCS is meant to be used with SVRG."
                                  " Please use get_lip_max instead.")

    @property
    def _epoch_size(self):
        return self._model.get_epoch_size()

    @property
    def _rand_max(self):
        return self._model.get_rand_max()
