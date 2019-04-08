# License: BSD 3 clause

from abc import abstractmethod
import numpy as np

from . import ModelFirstOrder

from .model import PASS_OVER_DATA, HESSIAN_NORM, N_CALLS_HESSIAN_NORM


class ModelSecondOrder(ModelFirstOrder):
    """An abstract class for models that implement a model with first
    order and second information, namely gradient and hessian norm
    information

    Attributes
    ----------
    n_calls_hessian_norm : `int` (read-only)
        Number of times ``hessian_norm`` has been called so far

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the data arrays used.

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """
    # A dict which specifies for each operation how many times we pass
    # through data
    pass_per_operation = {
        k: v
        for d in [ModelFirstOrder.pass_per_operation, {
            HESSIAN_NORM: 1
        }] for k, v in d.items()
    }

    _attrinfos = {N_CALLS_HESSIAN_NORM: {"writable": False}}

    def __init__(self):
        ModelFirstOrder.__init__(self)
        setattr(self, N_CALLS_HESSIAN_NORM, 0)

    def fit(self, *args):
        ModelFirstOrder.fit(self, *args)
        self._set(N_CALLS_HESSIAN_NORM, 0)
        return self

    def hessian_norm(self, coeffs: np.ndarray, point: np.ndarray) -> float:
        """Computes the norm given by coeffs^top * H * coeffs here H
        is the hessian of the model computed at ``point``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Vector where the hessian norm is computed

        point : `numpy.ndarray`
            Vector where we compute the hessian of the model

        Returns
        -------
        output : `float`
            Value of the hessian norm
        """
        if not self._fitted:
            raise Exception(
                "Must must fit data before calling ``hessian_norm`` ")
        if len(coeffs) != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 "expects %i coefficients") % (len(coeffs), self.n_coeffs))
        if len(point) != self.n_coeffs:
            raise ValueError(
                ("``point`` has size %i while the model" +
                 "expects %i coefficients") % (len(coeffs), self.n_coeffs))
        self._inc_attr(N_CALLS_HESSIAN_NORM)
        self._inc_attr(PASS_OVER_DATA,
                       step=self.pass_per_operation[HESSIAN_NORM])
        return self._hessian_norm(coeffs, point)

    @abstractmethod
    def _hessian_norm(self, coeffs: np.ndarray, point: np.ndarray) -> float:
        """Computes the norm given by form coeffs^top * H * coeffs
        here H is the Hessian of the model computed at ``point``

        Notes
        -----
        Must be overloaded in child class
        """
        pass
