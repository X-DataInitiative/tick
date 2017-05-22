# License: BSD 3 clause

import numpy as np
from .base import Prox
from .build.prox import ProxBinarsity as _ProxBinarsity


class ProxBinarsity(Prox):
    """Proximal operator of binarsity. It is simply a succession of two steps on
    different intervals: ``ProxTV`` plus a centering translation. More
    precisely, total-variation regularization is applied on a coefficient vector
    being a concatenation of multiple coefficient vectors corresponding to
    blocks, followed by centering within sub-blocks. Blocks (non-overlapping)
    are specified by the ``blocks_start`` and ``blocks_length`` parameters.

    Parameters
    ----------
    strength : `float`, default=0.
        Level of total-variation penalization

    blocks_start : `np.array`, shape=(n_blocks,)
        First entry of each block

    blocks_length : `np.array`, shape=(n_blocks,)
        Size of each block

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    n_blocks : `int`
        Number of blocks

    References
    ----------
    ProxBinarsity uses the fast-TV algorithm described in:

    Condat, L. (2012).
    `A Direct Algorithm for 1D Total Variation Denoising`_.

    .. _A Direct Algorithm for 1D Total Variation Denoising: https://hal.archives-ouvertes.fr/hal-00675043v2/document
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

    def __init__(self, strength: float, blocks_start: np.array,
                 blocks_length: np.array, range: tuple = None,
                 positive: bool = False):
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
        if any(blocks_start[1:] < blocks_start[:-1]):
            raise ValueError('``block_start`` must be sorted')

        if any(blocks_start[1:] < blocks_start[:-1] + blocks_length[:-1]):
            raise ValueError("blocks must not overlap")

        self.strength = strength
        self.positive = positive
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length

        if range is None:
            self._prox = _ProxBinarsity(strength, blocks_start,
                                        blocks_length, positive)
        else:
            start, end = self.range
            i_max = blocks_start.argmax()
            if end - start < blocks_start[i_max] + blocks_length[i_max]:
                raise ValueError("last block is not within the range "
                                 "[0, end-start)")
            self._prox = _ProxBinarsity(strength, blocks_start,
                                        blocks_length, start, end,
                                        positive)

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
