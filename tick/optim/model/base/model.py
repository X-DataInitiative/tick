# License: BSD 3 clause

from abc import ABC, abstractmethod
import numpy as np
from tick.base import Base

__author__ = 'Stephane Gaiffas'


LOSS = "loss"
GRAD = "grad"
LOSS_AND_GRAD = "loss_and_grad"
HESSIAN_NORM = "hessian_norm"

N_CALLS_LOSS = "n_calls_loss"
N_CALLS_GRAD = "n_calls_grad"
N_CALLS_LOSS_AND_GRAD = "n_calls_loss_and_grad"
N_CALLS_HESSIAN_NORM = "n_calls_hessian_norm"
PASS_OVER_DATA = "n_passes_over_data"


class Model(ABC, Base):
    """Abstract class for a model. It describes a zero-order model,
    namely only with the ability to compute a loss (goodness-of-fit
    criterion).

    Attributes
    ----------
    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_calls_loss : `int` (read-only)
        Number of times ``loss`` has been called so far

    n_passes_over_data : `int` (read-only)
        Number of effective passes through the data

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    # A dict which specifies for each operation how many times we
    # pass through data
    pass_per_operation = {LOSS: 1}

    _attrinfos = {
        "_fitted": {
            "writable": False
        },
        N_CALLS_LOSS: {
            "writable": False
        },
        PASS_OVER_DATA: {
            "writable": False
        },
        "n_coeffs": {
            "writable": False
        },
        "_model": {
            "writable": False
        }
    }

    # The name of the attribute that might contain the C++ model object
    _cpp_obj_name = "_model"

    def __init__(self):
        Base.__init__(self)
        self._fitted = False
        self._model = None
        setattr(self, N_CALLS_LOSS, 0)
        setattr(self, PASS_OVER_DATA, 0)

    def fit(self, *args):
        self._set_data(*args)
        self._set("_fitted", True)
        self._set(N_CALLS_LOSS, 0)
        self._set(PASS_OVER_DATA, 0)
        return self

    @abstractmethod
    def _get_n_coeffs(self) -> int:
        """An abstract method that forces childs to be able to give
        the number of parameters
        """
        pass

    @property
    def n_coeffs(self):
        if not self._fitted:
            raise ValueError(("call ``fit`` before using "
                              "``n_coeffs``"))
        return self._get_n_coeffs()

    @abstractmethod
    def _set_data(self, *args):
        """Must be overloaded in child class. This method is called to
        fit data onto the gradient.
        Useful when pre-processing is necessary, etc...
        """
        pass

    def loss(self, coeffs: np.ndarray) -> float:
        """Computes the value of the goodness-of-fit at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            The loss is computed at this point

        Returns
        -------
        output : `float`
            The value of the loss

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``loss``. An error is raised otherwise.
        """
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``loss``")
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(("``coeffs`` has size %i while the model" +
                              " expects %i coefficients") %
                             (len(coeffs), self.n_coeffs))
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(PASS_OVER_DATA,
                       step=self.pass_per_operation[LOSS])
        return self._loss(coeffs)

    @abstractmethod
    def _loss(self, coeffs: np.ndarray) -> float:
        """Must be overloaded in child class
        """
        pass
