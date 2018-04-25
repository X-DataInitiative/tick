# License: BSD 3 clause

import warnings
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

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

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
        self.dtype = None

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
            raise ValueError(("call ``fit`` before using " "``n_coeffs``"))
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
        # This is a bit of a hack as I don't see how to control the dtype of
        #  coeffs returning from scipy through lambdas
        if coeffs.dtype != self.dtype:
            warnings.warn(
                'coeffs vector of type {} has been cast to {}'.format(
                    coeffs.dtype, self.dtype))
            coeffs = coeffs.astype(self.dtype)

        if not self._fitted:
            raise ValueError("call ``fit`` before using ``loss``")
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(("``coeffs`` has size %i while the model" +
                              " expects %i coefficients") % (len(coeffs),
                                                             self.n_coeffs))
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[LOSS])

        return self._loss(coeffs)

    @abstractmethod
    def _loss(self, coeffs: np.ndarray) -> float:
        """Must be overloaded in child class
        """
        pass

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        """Deduce dtype and return true if C++ _model should be set
        """
        import six
        should_update_model = self._model is None
        local_dtype = None
        if (isinstance(dtype_or_object_with_dtype, six.string_types)
                or isinstance(dtype_or_object_with_dtype, np.dtype)):
            local_dtype = np.dtype(dtype_or_object_with_dtype)
        elif hasattr(dtype_or_object_with_dtype, 'dtype'):
            local_dtype = np.dtype(dtype_or_object_with_dtype.dtype)
        else:
            raise ValueError(("""
             unsupported type used for model creation,
             expects dtype or class with dtype , type:
             """ + self.__class__.__name__).strip())
        if self.dtype != local_dtype:
            should_update_model = True
        self.dtype = local_dtype
        if np.dtype(self.dtype) not in dtype_map:
            raise ValueError("""dtype does not exist in
              type map for """ + self.__class__.__name__.strip())
        return (should_update_model, dtype_map[np.dtype(self.dtype)])

    def astype(self, dtype_or_object_with_dtype):
        new_model = self._build_cpp_model(dtype_or_object_with_dtype)
        print("new_model", new_model)
        if new_model is not None:
            self._set('_model', new_model)
        return self

    def _build_cpp_model(self, dtype: str):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())
