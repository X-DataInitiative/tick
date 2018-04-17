# License: BSD 3 clause

from tick.prox.base import Prox
import numpy as np

from .build.prox import ProxSlopeDouble as _ProxSlopeDouble
from .build.prox import ProxSlopeFloat as _ProxSlopeFloat

dtype_map = {
    np.dtype("float64"): _ProxSlopeDouble,
    np.dtype("float32"): _ProxSlopeFloat
}


class ProxSlope(Prox):
    """Proximal operator of Slope penalization.
    This penalization is particularly relevant for feature selection, in
    generalized linear models, when features correlation is not too high.

    Parameters
    ----------
    strength : `float`
        Level of penalization

    fdr : `float`, default=0.6
        Desired False Discovery Rate for detection of non-zeros in
        the coefficients.
        Must be between 0 and 1.

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
            "cpp_setter": "set_false_discovery_rate"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        },
        "weights": {
            "writable": False,
        }
    }

    def __init__(self, strength: float, fdr: float = 0.6, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.strength = strength
        self.fdr = fdr
        self.positive = positive
        self.weights = None
        self._check_set_prox(dtype="float64")

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None):
        if Prox._check_set_prox(self, coeffs, dtype):
            if self.range is None:
                self._prox = dtype_map[np.dtype(self.dtype)](self.strength, self.fdr,
                                                   self.positive)
            else:
                self._prox = dtype_map[np.dtype(self.dtype)](
                    self.strength, self.fdr, self.range[0], self.range[1],
                    self.positive)

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
