# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox

from .build.prox import ProxPositiveDouble as _ProxPositiveDouble
from .build.prox import ProxPositiveFloat as _ProxPositiveFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxPositiveDouble,
    np.dtype("float32"): _ProxPositiveFloat
}


class ProxPositive(Prox):
    """Projection operator onto the half-space of vectors with
    non-negative entries

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector
    """

    def __init__(self, range: tuple = None, positive: bool = False):
        Prox.__init__(self, range)
        self._check_set_prox(dtype="float64")

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[np.dtype(self.dtype)](0.)
            else:
                self._prox = dtype_map[np.dtype(self.dtype)](0., self.range[0],
                                                   self.range[1])

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the projected ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            Vector to be projected

        Returns
        -------
        output : `float`
            Returns 0 (as this is a projection)
        """
        return self._prox.value(coeffs)
