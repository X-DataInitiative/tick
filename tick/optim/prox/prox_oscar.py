# License: BSD 3 clause

from tick.optim.prox.base import Prox
import numpy as np

from tick.optim.prox.build.prox import ProxOscar as _ProxOscar


class ProxOscar(Prox):
    """Proximal operator of the OSCAR penalization.
    This penalization combines L1 penalization with a clustering penalization,
    that induces exact equality of weights corresponding to correlated features,
    so that clusters can be represented by a single coefficient.
    This penalization is therefore particularly relevant in high-dimensional
    problems with strong features correlation.

    Parameters
    ----------
    strength : `float`
        Level of penalization

    ratio : `float`
        The Oscar ratio parameter, with ratio >= 0. For ratio = 0 this is L1
        regularization, while a large ratio provides only the clustering effect.

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply an extra projection onto the set of vectors with
        non-negative entries

    Notes
    -----
    This penalization was introduced in
    * Simultaneous regression shrinkage, variable selection and clustering of
      predictors with OSCAR, by Bondell H.D. and Reich B.J., Biometrics. 2008

    It uses the stack-based algorithm for FastProxL1 from

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

    def __init__(self, strength: float, ratio: float, range: tuple=None,
                 positive: bool=False):
        Prox.__init__(self, range)
        self.strength = strength
        self.ratio = ratio
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
