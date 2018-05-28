# License: BSD 3 clause

import numpy as np
from abc import ABC, abstractmethod
from tick.base import Base

deep_copy_ignore_fields = ["_prox"]


class Prox(ABC, Base):
    """An abstract base class for a proximal operator

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {"_prox": {"writable": False}, "_range": {"writable": False}}

    # The name of the attribute that will contain the C++ prox object
    _cpp_obj_name = "_prox"

    def __init__(self, range: tuple = None):
        Base.__init__(self)
        self._range = None
        self._prox = None
        self.range = range
        self.dtype = None

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        if val is not None:
            if len(val) != 2:
                raise ValueError("``range`` must be a tuple with 2 "
                                 "elements")
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

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)

    def _extract_dtype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.extract_dtype(
            dtype_or_object_with_dtype)

    def astype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        new_prox = tick.base.dtype_to_cpp_type.copy_with(
            self,
            ["_prox"]  # ignore _prox on deepcopy
        )
        new_prox._set('_prox',
                      new_prox._build_cpp_prox(dtype_or_object_with_dtype))
        return new_prox

    def _build_cpp_prox(self, dtype):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())
