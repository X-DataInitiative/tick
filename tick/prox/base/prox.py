# License: BSD 3 clause

import numpy as np
from abc import ABC, abstractmethod
from tick.base import Base


class Prox(ABC, Base):
    """An abstract base class for a proximal operator

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied
    """

    _attrinfos = {
        "_prox": {
            "writable": True
        },
        "_range": {
            "writable": False
        },
        "dtype": {
            "writable": True
        }
    }

    # The name of the attribute that will contain the C++ prox object
    _cpp_obj_name = "_prox"

    _allowable_exceptions = ["fmin_bfgs"]

    def __init__(self, range: tuple = None):
        Base.__init__(self)
        self.dtype = None
        self._range = None
        self._prox = None
        self.range = range

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        if val is not None:
            if len(val) != 2:
                raise ValueError("``range`` must be a tuple with 2 " "elements")
            if val[0] >= val[1]:
                raise ValueError("first element must be smaller than "
                                 "second element in ``range``")
            self._set("_range", val)
            _prox = self._prox
            if _prox is not None:
                _prox.set_start_end(val[0], val[1])

    def call(self, coeffs, step=1., out=None):
        """Apply proximal operator on a vector.
        It computes:

        .. math::
            argmin_x \\big( f(x) + \\frac{1}{2} \|x - v\|_2^2 \\big)

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            Input vector on which is applied the proximal operator

        step : `float` or `np.array`, default=1.
            The amount of penalization is multiplied by this amount

            * If `float`, the amount of penalization is multiplied by
              this amount
            * If `np.array`, then each coordinate of coeffs (within
              the given range), receives an amount of penalization
              multiplied by t (available only for separable prox)

        out : `numpy.ndarray`, shape=(n_params,), default=None
            If not `None`, the output is stored in the given ``out``.
            Otherwise, a new vector is created.

        Returns
        -------
        output : `numpy.ndarray`, shape=(n_coeffs,)
            Same object as out

        Notes
        -----
        ``step`` must have the same size as ``coeffs`` whenever range is
        `None`, or a size matching the one given by the range
        otherwise
        """
        if out is None:
            # We don't have an output vector, we create a fresh copy
            out = coeffs.copy()
        else:
            # We do an inplace copy of coeffs into out
            out[:] = coeffs
        # Apply the proximal, the output is in out
        self._call(coeffs, step, out)
        return out

    @abstractmethod
    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        pass

    @abstractmethod
    def value(self, coeffs: np.ndarray) -> float:
        pass

    def _check_set_prox(self, coeffs: np.ndarray = None, dtype=None) -> bool:
        if coeffs is None and dtype is None:
            raise ValueError("Method requires either ndarray or dtype")
        if coeffs is not None:
            dtype = coeffs.dtype
        ret = self.dtype is None or self.dtype != dtype
        self.dtype = dtype
        return ret

    def _check_stack_or_raise(self):
        import traceback
        acceptable = 0
        # this is a hack for python 3.4 which returns a list in "extract_stack"
        stack = traceback.extract_stack()
        if type(stack) is not list:
          stack = stack.format()
        for st in stack:
            for allowable in self._allowable_exceptions:
                if allowable in st:
                    acceptable = 1
                    break
            if acceptable == 1:
                break
        if acceptable == 0:
            raise ValueError("Stack check failure please sanitize your inputs")
