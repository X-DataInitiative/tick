# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxElasticNetDouble as _ProxElasticNet_d
from .build.prox import ProxElasticNetFloat as _ProxElasticNet_f

__author__ = 'Maryan Morel'

dtype_map = {
    np.dtype("float64"): _ProxElasticNet_d,
    np.dtype("float32"): _ProxElasticNet_f
}


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


    Attributes
    ----------
    strength : `float`
        Level of ElasticNet regularization

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

    def __init__(self,
                 strength: float,
                 ratio: float,
                 range: tuple = None,
                 positive=False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self.ratio = ratio
        self._check_set_prox(dtype=np.dtype("float64"))

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[self.dtype](self.strength, self.ratio,
                                                   self.positive)
            else:
                self._prox = dtype_map[self.dtype](self.strength, self.ratio,
                                                   self.range[0], self.range[1],
                                                   self.positive)

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
