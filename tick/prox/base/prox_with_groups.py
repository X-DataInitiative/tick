# License: BSD 3 clause

import numpy as np
from . import Prox


class ProxWithGroups(Prox):
    """Base class of a proximal operator with groups. It applies specific
    proximal operator in each group, or block. Blocks (non-overlapping) are
    specified by the ``blocks_start`` and ``blocks_length`` parameters.
    This base class is not intented for end-users, but for developers only.

    Parameters
    ----------
    strength : `float`
        Level of penalization

    blocks_start : `list` or `numpy.array`, shape=(n_blocks,)
        First entry of each block

    blocks_length : `list` or `numpy.array`, shape=(n_blocks,)
        Size of each block

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply the penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    n_blocks : `int`
        Number of blocks
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        },
        "blocks_start": {
            "writable": True,
            "cpp_setter": "set_blocks_start"
        },
        "blocks_length": {
            "writable": True,
            "cpp_setter": "set_blocks_length"
        }
    }

    def __init__(self, strength: float, blocks_start, blocks_length,
                 range: tuple = None, positive: bool = False):
        Prox.__init__(self, range)

        if any(length <= 0 for length in blocks_length):
            raise ValueError("all blocks must be of positive size")
        if any(start < 0 for start in blocks_start):
            raise ValueError("all blocks must have positive starting indices")

        if type(blocks_start) is list:
            blocks_start = np.array(blocks_start, dtype=np.uint64)
        if type(blocks_length) is list:
            blocks_length = np.array(blocks_length, dtype=np.uint64)

        if blocks_start.dtype is not np.uint64:
            blocks_start = blocks_start.astype(np.uint64)
        if blocks_length.dtype is not np.uint64:
            blocks_length = blocks_length.astype(np.uint64)

        if blocks_start.shape != blocks_length.shape:
            raise ValueError("``blocks_start`` and ``blocks_length`` "
                             "must have the same size")
        if np.any(blocks_start[1:] < blocks_start[:-1]):
            raise ValueError('``block_start`` must be sorted')
        if np.any(blocks_start[1:] < blocks_start[:-1] + blocks_length[:-1]):
            raise ValueError("blocks must not overlap")

        self.strength = strength
        self.positive = positive
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length

        # Get the C++ prox class, given by an overloaded method
        self._prox = self._build_cpp_prox("float64")

    @property
    def n_blocks(self):
        return self.blocks_start.shape[0]

    def _call(self, coeffs: np.ndarray, t: float, out: np.ndarray):
        self._prox.call(coeffs, t, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.array`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _as_dict(self):
        dd = Prox._as_dict(self)
        del dd["blocks_start"]
        del dd["blocks_length"]
        return dd

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        dtype_map = self._get_dtype_map()
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)

        if self.range is None:
            return prox_class(self.strength, self.blocks_start,
                              self.blocks_length, self.positive)
        else:
            start, end = self.range
            i_max = self.blocks_start.argmax()
            if end - start < self.blocks_start[i_max] + self.blocks_length[i_max]:
                raise ValueError("last block is not within the range "
                                 "[0, end-start)")
            return prox_class(self.strength, self.blocks_start,
                              self.blocks_length, start, end, self.positive)

    def _get_dtype_map(self):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())
