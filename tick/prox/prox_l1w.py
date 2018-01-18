# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxL1w as _ProxL1w

__author__ = 'Stephane Gaiffas'


# TODO: if we set a weights vector with length != end - start ???


class ProxL1w(Prox):
    """Proximal operator of the weighted L1 norm (weighted
    soft-thresholding)

    Parameters
    ----------
    strength : `float`
        Level of L1 penalization

    weights : `numpy.ndarray`, shape=(n_coeffs,)
        The weights to be used in the L1 penalization

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
        "weights": {
            "writable": True,
            "cpp_setter": "set_weights"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, weights: np.ndarray,
                 range: tuple=None, positive: bool=False):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxL1w(strength, weights, positive)
        else:
            start, end = range
            if (end - start) != weights.shape[0]:
                raise ValueError("Size of ``weights`` does not match "
                                 "the given ``range``")
            self._prox = _ProxL1w(strength, weights, range[0], range[1],
                                  positive)
        self.positive = positive
        self.strength = strength
        self.weights = weights

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

    def _as_dict(self):
        dd = Prox._as_dict(self)
        del dd["weights"]
        return dd
