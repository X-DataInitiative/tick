# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxElasticNetDouble as _ProxElasticNetDouble
from .build.prox import ProxElasticNetFloat as _ProxElasticNetFloat

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

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
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

    _cpp_class_dtype_map = {
        np.dtype("float64"): _ProxElasticNetDouble,
        np.dtype("float32"): _ProxElasticNetFloat
    }

    def __init__(self, strength: float, ratio: float, range: tuple = None,
                 positive=False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self.ratio = ratio
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
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           self._cpp_class_dtype_map)
        if self.range is None:
            return prox_class(self.strength, self.ratio, self.positive)
        else:
            return prox_class(self.strength, self.ratio, self.range[0],
                              self.range[1], self.positive)

    def _get_params_set(self):
        """Get the set of parameters
        """
        return {*Prox._get_params_set(self), 'strength', 'range', 'ratio'}

    @property
    def _AtomicClass(self):
        return AtomicProxElasticNet


from .build.prox import ProxElasticNetAtomicDouble as _ProxElasticNetAtomicDouble
from .build.prox import ProxElasticNetAtomicFloat as _ProxElasticNetAtomicFloat


class AtomicProxElasticNet(ProxElasticNet):
    _cpp_class_dtype_map = {
        np.dtype('float32'): _ProxElasticNetAtomicFloat,
        np.dtype('float64'): _ProxElasticNetAtomicDouble
    }

    def value(self, coeffs):
        return self._non_atomic_prox.value(coeffs)