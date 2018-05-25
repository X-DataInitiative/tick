# License: BSD 3 clause

import numpy as np
import scipy.sparse as sps
from tick.preprocessing.base import LongitudinalPreprocessor
from tick.preprocessing.build.preprocessing import LongitudinalFeaturesLagger\
    as _LongitudinalFeaturesLagger
from tick.preprocessing.utils import check_longitudinal_features_consistency,\
    check_censoring_consistency
from multiprocessing.pool import Pool
from copy import deepcopy
from functools import partial, partialmethod


class LongitudinalFeaturesLagger(LongitudinalPreprocessor):
    """Transforms longitudinal exposure features to add columns representing
    lagged features.

    This preprocessor transform an input list of `n_cases` numpy ndarrays or
    scipy.sparse.csr_matrices of shape `(n_intervals, n_features)` so as to
    add columns representing the lagged exposures. It outputs a list of
    `n_cases` numpy arrays or  csr_matrices of shape
    `(n_intervals, n_features + sum(n_lags + 1))`.

    Exposure can take two forms:
    - short repeated exposures: in that case, each column of the numpy arrays
    or csr matrices can contain multiple ones, each one representing an exposure
    for a particular time bucket.
    - infinite unique exposures: in that case, each column of the numpy arrays
    or csr matrices can only contain a single one, corresponding to the starting
    date of the exposure.

    Parameters
    ----------
    n_lags : `numpy.ndarray`, shape=(n_features,), dtype="uint64"
        Number of lags to compute: the preprocessor adds columns representing
        lag = 1, ..., n_lags[i] for each feature [i]. If `n_lags` is a null
        vector, this preprocessor does nothing. `n_lags` must be non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from tick.preprocessing.longitudinal_features_lagger import LongitudinalFeaturesLagger
    >>> features = [csr_matrix([[0, 1, 0],
    ...                         [0, 0, 0],
    ...                         [0, 0, 1]], dtype="float64"),
    ...             csr_matrix([[1, 1, 0],
    ...                         [0, 0, 1],
    ...                         [0, 0, 0]], dtype="float64")
    ...             ]
    >>> censoring = np.array([3, 2], dtype="uint64")
    >>> n_lags = np.array([2, 1, 0], dtype='uint64')
    >>> lfl = LongitudinalFeaturesLagger(n_lags)
    >>> product_features = lfl.fit_transform(features)
    >>> # output comes as a list of sparse matrices or 2D numpy arrays
    >>> product_features.__class__
    <class 'list'>
    >>> [x.toarray() for x in product_features]
    [array([[ 0.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.]]),
     array([[ 1.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.]])]
    """

    _attrinfos = {
        "n_lags": {
            "writable": False
        },
        "_n_init_features": {
            "writable": False
        },
        "_n_output_features": {
            "writable": False
        },
        "_n_intervals": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        }
    }

    def __init__(self, n_lags, n_jobs=-1):
        LongitudinalPreprocessor.__init__(self, n_jobs=n_jobs)
        if not isinstance(n_lags, np.ndarray) or n_lags.dtype != 'uint64':
            raise ValueError(
                '`n_lags` should be a numpy array of dtype uint64')
        self.n_lags = n_lags
        self._n_init_features = None
        self._n_output_features = None
        self._n_intervals = None
        self._fitted = False

    def _reset(self):
        """Resets the object its initial construction state."""
        self._set("_n_init_features", None)
        self._set("_n_output_features", None)
        self._set("_n_intervals", None)
        self._set("_fitted", False)

    def fit(self, features, labels=None, censoring=None):
        """Fit the feature lagger using the features matrices list.

        Parameters
        ----------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        censoring : `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.

        Returns
        -------
        output : `LongitudinalFeaturesLagger`
            The fitted current instance.
        """
        self._reset()
        base_shape = features[0].shape
        if base_shape[1] != len(self.n_lags):
            raise ValueError('Number of columns from feature matrices differs'
                             'from self.n_lags length.')
        features = check_longitudinal_features_consistency(
            features, base_shape, "float64")
        n_intervals, n_init_features = base_shape
        self._set("_n_init_features", n_init_features)
        self._set("_n_intervals", n_intervals)
        self._set("_n_output_features", int((self.n_lags + 1).sum()))
        self._set("_fitted", True)
        return self

    def transform(self, features, labels=None, censoring=None):
        """Add lagged features to the given features matrices list.

        Parameters
        ----------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        censoring : `numpy.ndarray`, shape=(n_cases,), dtype='uint64', default='None'
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the correponding matrix. The last n_intervals - c rows
            are then set to 0.

        Returns
        -------
        output : `[numpy.ndarrays]`  or `[csr_matrices]`, shape=(n_intervals, n_features)
            The list of features matrices with added lagged features.
        """
        n_samples = len(features)
        if censoring is None:
            censoring = np.full((n_samples,), self._n_intervals,
                                dtype="uint64")
        censoring = check_censoring_consistency(censoring, n_samples)
        base_shape = (self._n_intervals, self._n_init_features)
        features = check_longitudinal_features_consistency(
            features, base_shape, "float64")

        initializer = partial(self._inject_cpp_object,
                              n_intervals=self._n_intervals,
                              n_lags=self.n_lags)
        callback = self._sparse_lagger if sps.issparse(features[0]) \
            else self._dense_lagger
        callback = partial(callback, n_intervals=self._n_intervals,
                           n_output_features=self._n_output_features,
                           n_lags=self.n_lags)

        with Pool(self.n_jobs, initializer=initializer) as pool:
            X_with_lags = pool.starmap(callback, zip(features, censoring))

        return X_with_lags, labels, censoring

    @staticmethod
    def _inject_cpp_object(n_intervals, n_lags):
        """Creates a global instance of the CPP preprocessor object.

        WARNING: to be used only as a multiprocessing.Pool initializer.
        In multiprocessing context, each process has its own namespace, so using
        global is not as bad as it seems. Still, it requires to proceed with
        caution.
        """
        global _cpp_preprocessor
        _cpp_preprocessor = _LongitudinalFeaturesLagger(n_intervals, n_lags)

    @staticmethod
    def _dense_lagger(feature_matrix, censoring_i, n_intervals,
                      n_output_features, n_lags):
        """Creates a lagged version of a dense matrixrepresenting longitudinal
        features."""
        global _cpp_preprocessor
        output = np.zeros((n_intervals, n_output_features), dtype="float64")
        _cpp_preprocessor.dense_lag_preprocessor(feature_matrix, output,
                                                 int(censoring_i))
        return output

    @staticmethod
    def _sparse_lagger(feature_matrix, censoring_i, n_intervals,
                       n_output_features, n_lags):
        """Creates a lagged version of a sparse matrix representing longitudinal
        features."""
        global _cpp_preprocessor
        coo = feature_matrix.tocoo()
        estimated_nnz = coo.nnz * int((n_lags + 1).sum())
        out_row = np.zeros((estimated_nnz,), dtype="uint64")
        out_col = np.zeros((estimated_nnz,), dtype="uint64")
        out_data = np.zeros((estimated_nnz,), dtype="float64")
        _cpp_preprocessor.sparse_lag_preprocessor(
            coo.row.astype("uint64"), coo.col.astype("uint64"), coo.data,
            out_row, out_col, out_data, int(censoring_i))
        return sps.csr_matrix((out_data, (out_row, out_col)),
                              shape=(n_intervals, n_output_features))
