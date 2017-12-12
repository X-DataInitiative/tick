# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxMulti as _ProxMulti

__author__ = 'Stephane Gaiffas'


class ProxMulti(Prox):
    """
    Multiple proximal operator. This allows to apply sequentially a list
    of proximal operators. This is convenient when one wants to apply different
    proximal operators on different parts of a vector.

    Parameters
    ----------
    proxs : `tuple` of `Prox`
        A tuple of prox operators to be applied successively.
    """

    _attrinfos = {
        "proxs": {
            "writable": False,
        }
    }

    def __init__(self, proxs: tuple):
        Prox.__init__(self, None)
        for prox in proxs:
            if not isinstance(prox, Prox):
                raise ValueError('%s is not a Prox' % prox.__class__.__name__)
            if not hasattr(prox, '_prox'):
                raise ValueError('%s cannot be used in ProxMulti' % prox.name)
            if prox._prox is None:
                raise ValueError('%s cannot be used in ProxMulti' % prox.name)

        _proxs = [prox._prox for prox in proxs]
        # strength of ProxMulti is 0., since it's not used
        self._prox = _ProxMulti(_proxs)
        # Replace the list by a tuple to forbid changes
        self.proxs = proxs

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
