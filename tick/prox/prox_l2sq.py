# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxL2SqDouble as _ProxL2SqDouble
from .build.prox import ProxL2SqFloat as _ProxL2SqFloat

__author__ = 'Stephane Gaiffas'

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
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    _cpp_class_dtype_map = {
        np.dtype("float64"): _ProxL2SqDouble,
        np.dtype("float32"): _ProxL2SqFloat
    }

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
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
            return prox_class(self.strength, self.positive)
        else:
            return prox_class(self.strength, self.range[0], self.range[1],
                              self.positive)

    def _get_params_set(self):
        """Get the set of parameters
        """
        return {*Prox._get_params_set(self), 'strength', 'range'}

    @property
    def _AtomicClass(self):
        return AtomicProxL2Sq


from .build.prox import ProxL2SqAtomicDouble as _ProxL2SqAtomicDouble
from .build.prox import ProxL2SqAtomicFloat as _ProxL2SqAtomicFloat


class AtomicProxL2Sq(ProxL2Sq):
    _cpp_class_dtype_map = {
        np.dtype('float32'): _ProxL2SqAtomicFloat,
        np.dtype('float64'): _ProxL2SqAtomicDouble
    }

    def value(self, coeffs):
        return self._non_atomic_prox.value(coeffs)
