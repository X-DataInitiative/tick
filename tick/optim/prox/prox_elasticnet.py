# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxElasticNet as _ProxElasticNet

__author__ = 'Maryan Morel'


class ProxElasticNet(Prox):
    """
    Proximal operator of the ElasticNet regularization.

    Parameters
    ----------
    strength : `float`
        Level of ElasticNet regularization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    ratio : `float`, default=0
        The ElasticNet mixing parameter, with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.

    positive : `bool`, default=`False`
        If True, apply the penalization together with a projection
        onto the set of vectors with non-negative entries
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "ratio": {
            "writable": True,
            "cpp_setter": "set_ratio"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, ratio: float, range: tuple=None,
                 positive=False):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxElasticNet(strength, ratio, positive)
        else:
            self._prox = _ProxElasticNet(strength, ratio, range[0],
                                         range[1], positive)
        self.positive = positive
        self.strength = strength
        self.ratio = ratio

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
