# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from .base import Prox
from .build.prox import ProxZero as _ProxZero


__author__ = 'Stephane Gaiffas'


class ProxZero(Prox):
    """Proximal operator of the null function (identity)

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    Notes
    -----
    Using ``ProxZero`` means no penalization is applied on the model.
    """

    def __init__(self, range: tuple=None):
        Prox.__init__(self, range)
        if range is None:
            self._prox = _ProxZero(0.)
        else:
            self._prox = _ProxZero(0., range[0], range[1])

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
