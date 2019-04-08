# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxL1wDouble as _ProxL1wDouble
from .build.prox import ProxL1wFloat as _ProxL1wFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxL1wDouble,
    np.dtype("float32"): _ProxL1wFloat
}

# TODO: if we set a weights vector with length != end - start ???


class ProxL1w(Prox):
    """Proximal operator of the weighted L1 norm (weighted
    soft-thresholding)

    Parameters
    ----------
    strength : `float`
        Level of L1 penalization

    weights : `numpy.ndarray`, shape=(n_coeffs,)
        The weights to be used in the L1 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "weights": {
            "writable": True,
            "cpp_setter": "set_weights"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, weights: np.ndarray,
                 range: tuple = None, positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self.weights = weights
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _as_dict(self):
        dd = Prox._as_dict(self)
        del dd["weights"]
        return dd

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)

        weights = self.weights.astype(self.dtype)
        if self.range is None:
            return_prox = prox_class(self.strength, weights, self.positive)
        else:
            start, end = self.range
            if (end - start) != self.weights.shape[0]:
                raise ValueError("Size of ``weights`` does not match "
                                 "the given ``range``")
            return_prox = prox_class(self.strength, weights, self.range[0],
                                     self.range[1], self.positive)
        return_prox.weights = weights
        return return_prox
