# License: BSD 3 clause

import numpy as np
import scipy.sparse as sps
from itertools import combinations
from copy import deepcopy
from scipy.special import comb
from joblib import Parallel, delayed
from tick.preprocessing.base import LongitudinalPreprocessor
from .build.preprocessing import SparseLongitudinalFeaturesProduct
from .utils import check_longitudinal_features_consistency


class LongitudinalFeaturesProduct(LongitudinalPreprocessor):
    """Transforms longitudinal exposure features to add the corresponding
    product features.

    This preprocessor transform an input list of `n_cases` numpy arrays or
    csr_matrices of shape `(n_intervals, n_features)` so as to add columns
    representing the product of combination of two features. It outputs a list
    of `n_cases` numpy arrays or  csr_matrices of shape `(n_intervals,
    n_features + comb(n_features, 2))`.

    Exposure can take two forms:
    - short repeated exposures: in that case, each column of the numpy arrays
    or csr matrices can contain multiple ones, each one representing an exposure
    for a particular time bucket.
    - infinite unique exposures: in that case, each column of the numpy arrays
    or csr matrices can only contain a single one, corresponding to the starting
    date of the exposure.

    Parameters
    ----------
    exposure_type : {'infinite', 'short'}, default='infinite'
        Either 'infinite' for infinite unique exposures or 'short' for short
        repeated exposures.

    n_jobs : `int`, default=-1
        Number of tasks to run in parallel. If set to -1, the number of tasks is
        set to the number of cores.

    Attributes
    ----------
    mapper : `dict`
        Map product features to column indexes of the resulting matrices.

    Examples
    --------
    >>> from pprint import pprint
    >>> from scipy.sparse import csr_matrix
    >>> from tick.preprocessing.longitudinal_features_product import LongitudinalFeaturesProduct
    >>> infinite_exposures = [csr_matrix([[0, 1, 0],
    ...                                   [0, 0, 0],
    ...                                   [0, 0, 1]], dtype="float64"),
    ...                       csr_matrix([[1, 1, 0],
    ...                                   [0, 0, 1],
    ...                                   [0, 0, 0]], dtype="float64")
    ...                       ]
    >>> lfp = LongitudinalFeaturesProduct(exposure_type="infinite")
    >>> product_features, _, _ = lfp.fit_transform(features)
    >>> # output comes as a list of sparse matrices or 2D numpy arrays
    >>> product_features.__class__
    <class 'list'>
    >>> [x.toarray() for x in product_features]
    [array([[0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 1.]]), array([[1., 1., 0., 1., 0., 0.],
           [0., 0., 1., 0., 1., 1.],
           [0., 0., 0., 0., 0., 0.]])]
    """

    _attrinfos = {
        "exposure_type": {
            "writable": False
        },
        "_mapper": {
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
        "_preprocessor": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        }
    }

    def __init__(self, exposure_type="infinite", n_jobs=-1):
        LongitudinalPreprocessor.__init__(self, n_jobs=n_jobs)
        if exposure_type not in ["infinite", "finite"]:
            raise ValueError("exposure_type should be either 'infinite' or\
             'finite', not %s" % exposure_type)
        self.exposure_type = exposure_type
        self._reset()

    def _reset(self):
        """Resets the object its initial construction state."""
        self._set("_n_init_features", None)
        self._set("_n_output_features", None)
        self._set("_n_intervals", None)
        self._set("_mapper", {})
        self._set("_preprocessor", None)
        self._set("_fitted", False)

    @property
    def mapper(self):
        """Get the mapping between the feature products and column indexes.

        Returns
        -------
        output : `dict`
            The column index - feature mapping.
        """
        if not self._fitted:
            raise ValueError(
                "cannot get mapper if object has not been fitted.")
        return deepcopy(self._mapper)

    def fit(self, features, labels=None, censoring=None):
        """Fit the feature product using the features matrices list.

        Parameters
        ----------
        features : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        Returns
        -------
        output : `LongitudinalFeaturesProduct`
            The fitted current instance.
        """
        self._reset()
        base_shape = features[0].shape
        features = check_longitudinal_features_consistency(
            features, base_shape, "float64")
        n_intervals, n_init_features = base_shape
        if n_init_features < 2:
            raise ValueError("There should be at least two features to compute\
                     product features.")
        self._set("_n_init_features", n_init_features)
        self._set("_n_intervals", n_intervals)
        comb_it = combinations(range(n_init_features), 2)
        mapper = {i + n_init_features: c for i, c in enumerate(comb_it)}
        self._set("_mapper", mapper)
        self._set("_n_output_features",
                  int(n_init_features + comb(n_init_features, 2)))

        if sps.issparse(features[0]) and self.exposure_type == "infinite":
            self._set("_preprocessor",
                      SparseLongitudinalFeaturesProduct(features))

        self._set("_fitted", True)

        return self

    def transform(self, features, labels=None, censoring=None):
        """Add the product features to the given features matrices list.

        Parameters
        ----------
        features : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        Returns
        -------
        output : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_new_features)
            The list of features matrices with added product features.
            n_new_features = n_features + comb(n_features, 2)
        """

        base_shape = (self._n_intervals, self._n_init_features)
        features = check_longitudinal_features_consistency(
            features, base_shape, "float64")
        if self.exposure_type == "finite":
            X_with_products = self._finite_exposure_products(features)
        elif self.exposure_type == "infinite":
            X_with_products = self._infinite_exposure_products(features)
        else:
            raise ValueError("exposure_type should be either 'infinite' or\
                         'finite', not %s" % self.exposure_type)

        return X_with_products, labels, censoring

    def _infinite_exposure_products(self, features):
        """Add product features to features in the infinite exposure case."""
        if sps.issparse(features[0]):
            X_with_products = [
                self._sparse_infinite_product(arr) for arr in features
            ]
            # TODO later: fix multiprocessing
            # X_with_products = Parallel(n_jobs=self.n_jobs)(
            #     delayed(self._sparse_infinite_product)(arr) for arr in features)
            # Should be done in C++
        else:
            raise ValueError("Infinite exposures should be stored in \
                    sparse matrices as this hypothesis induces sparsity in the \
                    feature matrix.")

        return X_with_products

    def _finite_exposure_products(self, features):
        """Add product features to features in the finite exposure case."""
        if sps.issparse(features[0]):
            X_with_products = Parallel(n_jobs=self.n_jobs)(
                delayed(self._sparse_finite_product)(arr) for arr in features)
        else:
            X_with_products = Parallel(n_jobs=self.n_jobs)(
                delayed(self._dense_finite_product)(arr) for arr in features)

        return X_with_products

    def _dense_finite_product(self, feat_mat):
        """Performs feature product on a numpy.ndarray containing
        finite exposures."""
        feat = [feat_mat]
        feat.extend([(feat_mat[:, i] * feat_mat[:, j]).reshape((-1, 1))
                     for i, j in self._mapper.values()])
        return np.hstack(feat)

    def _sparse_finite_product(self, feat_mat):
        """Performs feature product on a scipy.sparse.csr_matrix containing
        finite exposures."""
        feat = [feat_mat.tocsc()]
        feat.extend([(feat_mat[:, i].multiply(feat_mat[:, j]))
                     for i, j in self.mapper.values()])
        return sps.hstack(feat).tocsr()

    def _sparse_infinite_product(self, feat_mat):
        """Performs feature product on a scipy.sparse.csr_matrix containing
        infinite exposures."""
        coo = feat_mat.tocoo()
        nnz = coo.nnz
        new_nnz = self._n_output_features * nnz
        new_row = np.zeros((new_nnz,), dtype="uint64")
        new_col = np.zeros((new_nnz,), dtype="uint64")
        new_data = np.zeros((new_nnz,), dtype="float64")
        self._preprocessor.sparse_features_product(
            coo.row.astype("uint64"), coo.col.astype("uint64"), coo.data,
            new_row, new_col, new_data)
        return sps.csr_matrix((new_data, (new_row, new_col)),
                              shape=(self._n_intervals,
                                     self._n_output_features))
