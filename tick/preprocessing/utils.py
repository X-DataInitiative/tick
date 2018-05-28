# License: BSD 3 clause

import numpy as np
import pandas as pd
from warnings import warn


def safe_array(X, dtype=np.float64):
    """Checks if the X has the correct type, dtype, and is contiguous.

    Parameters
    ----------
    X : `pd.DataFrame` or `np.ndarray` or `crs_matrix`
        The input data.

    dtype : np.dtype object or string
        Expected dtype of each X element.

    Returns
    -------
    output : `np.ndarray` or `csr_matrix`
        The input with right type, dtype.

    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(X, np.ndarray) and not X.flags['C_CONTIGUOUS']:
        warn(
            'Copying array of size %s to create a C-contiguous '
            'version of it' % str(X.shape), RuntimeWarning)
        X = np.ascontiguousarray(X)

    if X.dtype != dtype:
        warn(
            'Copying array of size %s to convert it in the right '
            'format' % str(X.shape), RuntimeWarning)
        X = X.astype(dtype)

    return X


def check_longitudinal_features_consistency(X, shape, dtype):
    """Checks if all elements of longitudinal features X have the same shape
     and correct dtype. In case the dtype is not correct, convert X elements
     to the proper dtypes.

    Parameters
    ----------
    X : list of np.ndarray or list of scipy.sparse.csr_matrix,
        list of length n_samples, each element of the list of
        shape=(n_intervals, n_features)
        The list of features matrices.

    shape : Tuple(`int`, `int`)
        Expected shape of each X element.

    dtype : np.dtype object or string
        Expected dtype of each X element.

    Returns
    -------
    output : list of np.ndarray or list of scipy.sparse.csr_matrix,
        list of length n_samples, each element of the list of
        shape=(n_intervals, n_features)
        The list of features matrices with corrected shapes and dtypes.
    """
    if not all([x.shape == shape for x in X]):
        raise ValueError("All the elements of X should have the same\
         shape.")
    return [safe_array(x, dtype) for x in X]


def check_censoring_consistency(censoring, n_samples):
    """Checks if `censoring` has the correct shape and dtype, which is 'uint64'.

    Parameters
    ----------
    censoring : `np.ndarray`, shape=(n_samples, 1), dtype="uint64"
        The censoring data.
        Cf. tick.preprocessing.LongitudinalFeaturesLagger

    n_samples : `int`
        The expected number of samples.

    Returns
    -------
    output : `np.ndarray`, shape=(n_samples, 1), dtype="uint64"
        The censoring data with right shape and dtype.

    """
    if censoring.shape != (n_samples,):
        raise ValueError("`censoring` should be a 1-D numpy ndarray of \
            shape (%i,)" % n_samples)
    return safe_array(censoring, "uint64")
