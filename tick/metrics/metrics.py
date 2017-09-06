# License: BSD 3 clause

import numpy as np


def support_fdp(x_truth, x, eps=1e-8):
    """Computes the False Discovery Proportion for selecting the support
    of x_truth using x, namely the proportion of false positive among all
    detected positives, given by FP / (FP + TP). This is useful to assess
    the features selection or outliers detection abilities of a learner.

    Parameters
    ----------
    x_truth : `numpy.array`
        Ground truth weights

    x : `numpy.array`
        Learned weights

    Returns
    -------
    output : `float`
        The False Discovery Proportion for detecting the support of ``x_truth``
        using the support of ``x``
    """
    s = np.abs(x) > eps
    s_truth = np.abs(x_truth) > eps
    v = np.logical_and(np.logical_not(s_truth), s).sum()
    r = max(s.sum(), 1)
    return v / r


def support_recall(x_truth, x, eps=1e-8):
    """Computes proportion of true positives (TP) among the number ground
    truth positives (namely TP + FN, where FN is the number of false
    negatives), hence TP / (TP + FN). This is useful to assess
    the features selection or outliers detection abilities of a learner.

    Parameters
    ----------
    x_truth : `numpy.array`
        Ground truth weights

    x : `numpy.array`
        Learned weights

    Returns
    -------
    output : `float`
        The False Discovery Proportion for detecting the support of ``x_truth``
        using the support of ``x``
    """
    s = np.abs(x) > eps
    s_truth = np.abs(x_truth) > eps
    v = np.logical_and(s_truth, s).sum()
    return v / s_truth.sum()
