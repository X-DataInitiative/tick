# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox

from .build.prox import ProxTVDouble as _ProxTVDouble
from .build.prox import ProxTVFloat as _ProxTVFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {np.dtype("float64"): _ProxTVDouble, np.dtype("float32"): _ProxTVFloat}


class ProxTV(Prox):
    """Proximal operator of the total-variation penalization

    Parameters
    ----------
    strength : `float`
        Level of total-variation penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
        onto the set of vectors with non-negative entries

    Notes
    -----
    Uses the fast-TV algorithm described in:

    * "A Direct Algorithm for 1D Total Variation Denoising"
      by Laurent Condat, *Ieee Signal Proc. Letters*
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self._check_set_prox(dtype="float64")

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[np.dtype(self.dtype)](self.strength,
                                                   self.positive)
            else:
                self._prox = dtype_map[np.dtype(self.dtype)](
                    self.strength, self.range[0], self.range[1], self.positive)

    def _call(self, coeffs: np.ndarray, step: float, out: np.ndarray):
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
