# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxPositive as _ProxPositive


__author__ = 'Stephane Gaiffas'


class ProxPositive(Prox):
    """Projection operator onto the half-space of vectors with
    non-negative entries

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector
    """

    def __init__(self, range: tuple=None, positive: bool=False):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxPositive(0.)
        else:
            self._prox = _ProxPositive(0., range[0], range[1])

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
