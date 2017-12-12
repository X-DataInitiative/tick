# License: BSD 3 clause

import numpy as np
from numpy.linalg import norm


def relative_distance(new_vector, old_vector, use_norm=None):
    """Computes the relative error with respect to some norm
    It is useful to evaluate relative change of a vector

    Parameters
    ----------
    new_vector : `np.ndarray`
        New value of the vector

    old_vector : `np.ndarray`
        old value of the vector to compare with

    use_norm : `int` or `str`
        The norm to use among those proposed by :func:`.np.linalg.norm`

    Returns
    -------
    output : `float`
        Relative distance
    """
    norm_old_vector = norm(old_vector, use_norm)
    if norm_old_vector == 0:
        norm_old_vector = 1.
    return norm(new_vector - old_vector, use_norm) / norm_old_vector
