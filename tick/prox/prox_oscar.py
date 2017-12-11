# License: BSD 3 clause

# -*- coding: utf8 -*-

from tick.prox.base import Prox
import numpy as np

from .build.prox import ProxSortedL1 as _ProxSortedL1
from .build.prox import WeightsType_bh, WeightsType_oscar

# TODO: put also the OSCAR weights
# TODO: we should be able to put any weights we want...


class ProxSortedL1(Prox):
    """Proximal operator of sorted L1 penalization

    Parameters
    ----------
    strength : `float`
        Level of penalization

    fdr : `float`, default=0.6
        Desired False Discovery Rate for detection of non-zeros in
        the coefficients

    weights_type : "bh" | "oscar", default="bh"

        * If "bh", we use Benjamini-Hochberg weights, under a Gaussian
          error assumption, and expect a FDR control
        * If "oscar", there is no FDR control, and we get the OSCAR
          penalization, see notes below for references

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    Attributes
    ----------
    weights : `np.array`, shape=(n_coeffs,)
        The weights used in the penalization. They are automatically
        setted, depending on the ``weights_type`` and ``fdr``
        parameters.

    Notes
    -----
    Uses the stack-based algorithm for FastProxL1 from

    * SLOPE--Adaptive Variable Selection via Convex Optimization, by
      Bogdan, M. and Berg, E. van den and Sabatti, C. and Su, W. and Candes, E. J.
      arXiv preprint arXiv:1407.3824, 2014
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "fdr": {
            "writable": True,
            "cpp_setter": "set_fdr"
        },
        "_weights_type": {
            "writable": False,
            "cpp_setter": "set_weights_type"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        },
        "weights": {
            "writable": False,
        }
    }

    def __init__(self, strength: float, fdr: float=0.6,
                 weights_type: str="bh", range: tuple=None,
                 positive: bool=False):
        Prox.__init__(self, range)
        self.strength = strength
        self.fdr = fdr
        self.weights_type = weights_type
        self.positive = positive
        self.weights = None
        if range is None:
            self._prox = _ProxSortedL1(self.strength, self.fdr,
                                       self._weights_type,
                                       self.positive)
        else:
            self._prox = _ProxSortedL1(self.strength, self.fdr,
                                       self._weights_type,
                                       self.range[0], self.range[1],
                                       self.positive)

    @property
    def weights_type(self):
        if self._weights_type == WeightsType_bh:
            return "bh"
        elif self._weights_type == WeightsType_oscar:
            return "oscar"

    @weights_type.setter
    def weights_type(self, val):
        if val == "bh":
            self._set("_weights_type", WeightsType_bh)
        elif val == "oscar":
            self._set("_weights_type", WeightsType_oscar)
            raise NotImplementedError("``oscar`` weights.")
        else:
            raise ValueError("``weights_type`` must be either 'bh' "
                             "or 'oscar'")

    def _call(self, coeffs: np.ndarray, t: float, out: np.ndarray):
        self._prox.call(coeffs, t, out)

    def value(self, coeffs: np.ndarray) -> float:
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
