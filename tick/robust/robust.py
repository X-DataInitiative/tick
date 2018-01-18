# License: BSD 3 clause

import numpy as np


def std_mad(x):
    """Robust estimation of the standard deviation, based on the Corrected Median
    Absolute Deviation (MAD) of x.
    This computes the MAD of x, and applies the Gaussian distribution
    correction, making it a consistent estimator of the standard-deviation
    (when the sample looks Gaussian with outliers).

    Parameters
    ----------
    x : `np.ndarray`
        Input vector

    Returns
    -------
    output : `float`
        A robust estimation of the standard deviation
    """
    from scipy.stats import norm
    correction = 1 / norm.ppf(3 / 4)
    return correction * np.median(np.abs(x - np.median(x)))


def std_iqr(x):
    """Robust estimation of the standard deviation, based on the inter-quartile
    (IQR) distance of x.
    This computes the IQR of x, and applies the Gaussian distribution
    correction, making it a consistent estimator of the standard-deviation
    (when the sample looks Gaussian with outliers).

    Parameters
    ----------
    x : `np.ndarray`
        Input vector

    Returns
    -------
    output : `float`
        A robust estimation of the standard deviation
    """
    from scipy.stats import iqr
    from scipy.special import erfinv

    correction = 2 ** 0.5 * erfinv(0.5)
    return correction * iqr(x)
