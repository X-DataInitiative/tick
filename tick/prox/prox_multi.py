# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxMultiDouble as _ProxMultiDouble
from .build.prox import ProxMultiFloat as _ProxMultiFloat
from tick.prox import ProxZero

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxMultiDouble,
    np.dtype("float32"): _ProxMultiFloat
}


class ProxMulti(Prox):
    """Multiple proximal operator. This allows to apply sequentially a list
    of proximal operators. This is convenient when one wants to apply different
    proximal operators on different parts of a vector.

    Parameters
    ----------
    proxs : `tuple` of `Prox`
        A tuple of prox operators to be applied successively.

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {"proxs": {"writable": False,}}

    def __init__(self, proxs: tuple):
        Prox.__init__(self, None)
        if not proxs:
            proxs = [ProxZero()]
        dtype = proxs[0].dtype
        self.dtype = dtype
        for prox in proxs:
            if not isinstance(prox, Prox):
                raise ValueError('%s is not a Prox' % prox.__class__.__name__)
            if not hasattr(prox, '_prox'):
                raise ValueError('%s cannot be used in ProxMulti' % prox.name)
            if prox._prox is None:
                raise ValueError('%s cannot be used in ProxMulti' % prox.name)
            if dtype != prox.dtype:
                raise ValueError(
                    'ProxMulti can only handle proxes with same dtype')

        # strength of ProxMulti is 0., since it's not used
        self.proxs = [prox._prox for prox in proxs]
        self._prox = self._build_cpp_prox(dtype)

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``.
        This returns the sum of the values of each prox called on the same
        coeffs.

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
                                           dtype_map)
        return prox_class(self.proxs)

    def astype(self, dtype_or_object_with_dtype):
        raise NotImplementedError(
            "This type requires each Prox to their 'astype' called (for now)")
