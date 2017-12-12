# License: BSD 3 clause

import numpy as np
from .base import Prox
from .build.prox import ProxL2 as _ProxL2


__author__ = 'Stephane Gaiffas'


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
            self._prox = _ProxL2(strength, positive)
        else:
            self._prox = _ProxL2(strength, range[0], range[1], positive)
        self.positive = positive
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
