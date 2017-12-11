# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxL1 as _ProxL1

__author__ = 'Stephane Gaiffas'


class ProxL1(Prox):
    """Proximal operator of the L1 norm (soft-thresholding)

    Parameters
    ----------
    strength : `float`
        Level of L1 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
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

    def __init__(self, strength: float, range: tuple=None,
                 positive: bool=False):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxL1(strength, positive)
        else:
            self._prox = _ProxL1(strength, range[0], range[1], positive)
        self.positive = positive
        self.strength = strength

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """Returns the value of the penalization at ``coeffs``

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
