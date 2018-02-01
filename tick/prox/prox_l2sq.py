# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxL2SqDouble as _ProxL2sqDouble
from .build.prox import ProxL2SqFloat as _ProxL2sqFloat

__author__ = 'Stephane Gaiffas'

dtype_map = {
  np.float64: _ProxL2sqDouble,
  np.float32: _ProxL2sqFloat
}

class ProxL2Sq(Prox):
    """Proximal operator of the squared L2 norm (ridge penalization)

    Parameters
    ----------
    strength : `float`, default=0.
        Level of L2 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L2 penalization together with a projection
        onto the set of vectors with non-negative entries
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

    def __init__(
        self, strength: float, range: tuple=None,
        positive: bool=False, dtype=np.float64
    ):
        Prox.__init__(self, range, dtype=dtype)
        if self.dtype not in dtype_map:
            raise ValueError('dtype provided to ProxL2Sq is not handled')
        if range is None:
            self._prox = dtype_map[self.dtype](strength, positive)            
        else:
            self._prox = dtype_map[self.dtype](strength, range[0], range[1], positive)
        self.positive = positive
        self.strength = strength

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
        if self.dtype != np.float64:
          coeffs = coeffs.astype(self.dtype)
        return self._prox.value(coeffs)
