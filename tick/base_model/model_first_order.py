# License: BSD 3 clause

import warnings
from abc import abstractmethod
import numpy as np

from . import Model
from .model import GRAD, LOSS_AND_GRAD, N_CALLS_GRAD, \
    N_CALLS_LOSS_AND_GRAD, PASS_OVER_DATA, N_CALLS_LOSS


class ModelFirstOrder(Model):
    """An abstract class for models that implement a model with first
    order information, namely gradient information

    Attributes
    ----------
    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_calls_loss : `int` (read-only)
        Number of times ``loss`` has been called so far

    n_passes_over_data : `int` (read-only)
        Number of effective passes through the data

    n_calls_grad : `int` (read-only)
        Number of times ``grad`` has been called so far

    n_calls_loss_and_grad : `int` (read-only)
        Number of times ``loss_and_grad`` has been called so far

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """
    # A dict which specifies for each operation how many times we pass
    # through data
    pass_per_operation = {
        k: v
        for d in [Model.pass_per_operation, {
            GRAD: 1,
            LOSS_AND_GRAD: 2
        }] for k, v in d.items()
    }

    _attrinfos = {
        N_CALLS_GRAD: {
            "writable": False
        },
        N_CALLS_LOSS_AND_GRAD: {
            "writable": False
        },
    }

    def __init__(self):
        Model.__init__(self)
        setattr(self, N_CALLS_GRAD, 0)
        setattr(self, N_CALLS_LOSS_AND_GRAD, 0)

    def fit(self, *args):
        Model.fit(self, *args)
        self._set(N_CALLS_GRAD, 0)
        self._set(N_CALLS_LOSS_AND_GRAD, 0)
        return self

    def grad(self, coeffs: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """Computes the gradient of the model at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Vector where gradient is computed

        out : `numpy.ndarray` or `None`
            If `None` a new vector containing the gradient is returned,
            otherwise, the result is saved in ``out`` and returned

        Returns
        -------
        output : `numpy.ndarray`
            The gradient of the model at ``coeffs``

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``grad``. An error is raised otherwise.
        """
        if coeffs.dtype != self.dtype:
            warnings.warn(
                'coeffs vector of type {} has been cast to {}'.format(
                    coeffs.dtype, self.dtype))
            coeffs = coeffs.astype(self.dtype)

        if not self._fitted:
            raise ValueError("call ``fit`` before using ``grad``")

        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 " expects %i coefficients") % (len(coeffs), self.n_coeffs))

        if out is not None:
            grad = out
        else:
            grad = np.empty(self.n_coeffs, dtype=self.dtype)

        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[GRAD])

        self._grad(coeffs, out=grad)
        return grad

    @abstractmethod
    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        """Computes the gradient of the model at ``coeffs``
        The gradient must be stored in ``out``

        Notes
        -----
        Must be overloaded in child class
        """
        pass

    # TODO: better method annotation giving the type in the tuple
    def loss_and_grad(self, coeffs: np.ndarray,
                      out: np.ndarray = None) -> tuple:
        """Computes the value and the gradient of the function at
        ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Vector where the loss and gradient are computed

        out : `numpy.ndarray` or `None`
            If `None` a new vector containing the gradient is returned,
            otherwise, the result is saved in ``out`` and returned

        Returns
        -------
        loss : `float`
            The value of the loss

        grad : `numpy.ndarray`
            The gradient of the model at ``coeffs``

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``loss_and_grad``. An error is raised otherwise.

        """
        if not self._fitted:
            raise ValueError("call ``fit`` before using " "``loss_and_grad``")

        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 "expects %i coefficients") % (len(coeffs), self.n_coeffs))
        if out is not None:
            grad = out
        else:
            grad = np.empty(self.n_coeffs, dtype=self.dtype)

        self._inc_attr(N_CALLS_LOSS_AND_GRAD)
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA,
                       step=self.pass_per_operation[LOSS_AND_GRAD])
        loss = self._loss_and_grad(coeffs, out=grad)
        return loss, grad

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray) -> float:
        self._grad(coeffs, out=out)
        return self._loss(coeffs)
