# License: BSD 3 clause

from tick.optim.prox.base import Prox
import numpy as np

from tick.optim.prox.build.prox import ProxOscar as _ProxOscar


class ProxOscar(Prox):
    """Proximal operator the oscar penalization.
    This penalization is particularly relevant for feature selection, in
    generalized linear models, when features correlation is not too high.

    Parameters
    ----------
    strength : `float`
        Level of penalization

    # TODO: clean this docstring
    ratio : `float`, default=0
        The Oscar mixing parameter, with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply an extra projection onto the set of vectors with
        non-negative entries

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
        "ratio": {
            "writable": True,
            "cpp_setter": "set_ratio"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, range: tuple=None,
                 positive: bool=False):
        Prox.__init__(self, range)
        self.strength = strength
        self.positive = positive
        self.weights = None
        if range is None:
            self._prox = _ProxOscar(self.strength, self.positive)
        else:
            self._prox = _ProxOscar(self.strength, self.range[0], self.range[1],
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
