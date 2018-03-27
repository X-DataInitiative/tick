# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
import sys

from .base import Prox
from .build.prox import ProxEqualityDouble as _ProxEquality_d
from .build.prox import ProxEqualityFloat as _ProxEquality_f

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxEquality_d,
    np.dtype("float32"): _ProxEquality_f
}


class ProxEquality(Prox):
    """Projection operator onto the set of vector with all coordinates equal
    (or in the given range if given one).
    Namely, this simply replaces all coordinates by their average

    Parameters
    ----------
    strength : `float`, default=0.
        Not used in this prox, but kept for compatibility issues

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, ensures that the output of the prox has only non-negative
        entries (in the given range)

    dtype : `string`
        Type of arrays to use - default float64

    """

    _attrinfos = {"positive": {"writable": True, "cpp_setter": "set_positive"}}

    def __init__(self,
                 strength: float = 0,
                 range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self._check_set_prox(dtype=np.dtype("float64"))

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[self.dtype](0., self.positive)
            else:
                self._prox = dtype_map[self.dtype](0., self.range[0],
                                                   self.range[1], self.positive)

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Simply returns 0 if all coeffs in range are equal. Other wise returns
        infinity. This is not a penalization but a projection.

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            Vector to be projected

        Returns
        -------
        output : `float`
            Returns 0 or np.inf
        """
        raw_value = self._prox.value(coeffs)
        if raw_value == sys.float_info.max:
            return np.inf
        else:
            return 0

    @property
    def strength(self):
        return None

    @strength.setter
    def strength(self, val):
        # Strength is not settable in this prox
        pass
