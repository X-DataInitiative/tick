# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
import sys

from .base import Prox
from .build.prox import ProxEquality as _ProxEquality

__author__ = 'Stephane Gaiffas'


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
    """

    _attrinfos = {
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float=0, range: tuple=None, positive: bool=False):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxEquality(0., positive)
        else:
            self._prox = _ProxEquality(0., range[0], range[1], positive)

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
