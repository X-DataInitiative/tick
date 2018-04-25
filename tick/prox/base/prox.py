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


    Attributes
    ----------

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

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
        """Apply proximal operator on a vector.
        Deduce dtype and return true if C++ _prox should be set
        """
        import six
        should_update_prox = False
        local_dtype = None
        if (isinstance(dtype_or_object_with_dtype, six.string_types)
                or isinstance(dtype_or_object_with_dtype, np.dtype)):
            local_dtype = np.dtype(dtype_or_object_with_dtype)
        elif hasattr(dtype_or_object_with_dtype, 'dtype'):
            local_dtype = np.dtype(dtype_or_object_with_dtype.dtype)
        else:
            raise ValueError(("""
             unsupported type used for prox creation,
             expects dtype or class with dtype , type:
             """ + self.__class__.__name__).strip())
        if self.dtype is None or self.dtype != local_dtype:
            should_update_prox = True
        self.dtype = local_dtype
        if np.dtype(self.dtype) not in dtype_map:
            raise ValueError("""dtype does not exist in
              type map for """ + self.__class__.__name__.strip())
        return (should_update_prox, dtype_map[np.dtype(self.dtype)])

    def astype(self, dtype_or_object_with_dtype):
        new_prox = self._build_cpp_prox(dtype_or_object_with_dtype)
        if new_prox is not None:
            self._set('_prox', new_prox)
        return self

    def _build_cpp_prox(self, dtype: str):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())
