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

    Notes
    -----
    Using ``ProxZero`` means no penalization is applied on the model.
    """

    def __init__(self, range: tuple = None):
        Prox.__init__(self, range)
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

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        (updated_prox, prox_class) = \
            self._get_typed_class(dtype_or_object_with_dtype, dtype_map)
        if updated_prox is True:
            if self.range is None:
                return prox_class(0.)
            else:
                return prox_class(0., self.range[0], self.range[1])
        return None
