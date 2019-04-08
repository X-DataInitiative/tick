# License: BSD 3 clause

import numpy as np
from .base import Prox
from .build.prox import ProxL2Double as _ProxL2Double
from .build.prox import ProxL2Float as _ProxL2Float

__author__ = 'Stephane Gaiffas'

dtype_map = {
    np.dtype("float64"): _ProxL2Double,
    np.dtype("float32"): _ProxL2Float
}


class ProxL2(Prox):
    """Proximal operator of the L2 penalization. Do not mix up with ProxL2sq,
    which is regular ridge (squared L2) penalization. ProxL2 induces sparsity
    on the full vector, whenever the norm of it is small enough.
    This is mostly used in the ProxGroupL1 for group-lasso penalization.

    Parameters
    ----------
    strength : `float`
        Level of penalization. Note that in this proximal operator, ``strength``
        is automatically multiplied by the square-root of ``end`` - ``start``,
        when a range is used, or ``n_coeffs``, when no range is used
        (size of the passed vector). This allows to consider strengths that have
        the same order as with `ProxL1` or other separable proximal operators.

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

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self._prox = self._build_cpp_prox("float64")

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

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        if self.range is None:
            return prox_class(self.strength, self.positive)
        else:
            return prox_class(self.strength, self.range[0], self.range[1],
                              self.positive)
