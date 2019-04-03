# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox

from .build.prox import ProxZeroDouble as _ProxZeroDouble
from .build.prox import ProxZeroFloat as _ProxZeroFloat

__author__ = 'Stephane Gaiffas'


class ProxZero(Prox):
    """Proximal operator of the null function (identity)

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.

    Notes
    -----
    Using ``ProxZero`` means no penalization is applied on the model.
    """
    _cpp_class_dtype_map = {
        np.dtype("float64"): _ProxZeroDouble,
        np.dtype("float32"): _ProxZeroFloat
    }

    def __init__(self, range: object = None) -> object:
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
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           self._cpp_class_dtype_map)
        if self.range is None:
            return prox_class(0.)
        else:
            return prox_class(0., self.range[0], self.range[1])


    @property
    def _AtomicClass(self):
        return AtomicProxZero


from .build.prox import ProxZeroAtomicDouble as _ProxZeroAtomicDouble
from .build.prox import ProxZeroAtomicFloat as _ProxZeroAtomicFloat


class AtomicProxZero(ProxZero):
    _cpp_class_dtype_map = {
        np.dtype('float32'): _ProxZeroAtomicFloat,
        np.dtype('float64'): _ProxZeroAtomicDouble
    }

