# -*- coding: utf8 -*-


import numpy as np
from .base import Prox
from .build.prox import ProxL1L2 as _ProxL1L2


__author__ = 'Stephane Gaiffas'


class ProxL1L2(Prox):
    """Proximal operator of the l1/l2 penalization

    Parameters
    ----------
    strength : `float`, default=0.
        Level of l1/l2 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    Notes
    -----
    Use this prox in combination with ProxMulti to obtain GroupLasso.
    Ranges should describe non overlapping groups.
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        }
    }

    def __init__(self, strength: float, range: tuple=None):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxL1L2(strength, False)
        else:
            self._prox = _ProxL1L2(strength, range[0], range[1], False)
        self.strength = strength

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
