# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox

from .build.prox import ProxZeroDouble as _ProxZeroDouble
from .build.prox import ProxZeroFloat as _ProxZeroFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxZeroDouble,
    np.dtype("float32"): _ProxZeroFloat
}


class ProxZero(Prox):
    """Proximal operator of the null function (identity)

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    dtype : `string`, default='float64'
        Type of arrays to use - default float64

    Notes
    -----
    Using ``ProxZero`` means no penalization is applied on the model.
    """

    def __init__(self, range: tuple = None):
        Prox.__init__(self, range)
        self._check_set_prox(dtype="float64")

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

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[np.dtype(self.dtype)](0.)
            else:
                self._prox = dtype_map[np.dtype(self.dtype)](0., self.range[0],
                                                   self.range[1])
