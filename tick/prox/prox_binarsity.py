# License: BSD 3 clause

from .base import ProxWithGroups
from .build.prox import ProxBinarsityDouble as _ProxBinarsityDouble
from .build.prox import ProxBinarsityFloat as _ProxBinarsityFloat

import numpy as np

dtype_map = {
    np.dtype("float64"): _ProxBinarsityDouble,
    np.dtype("float32"): _ProxBinarsityFloat
}


class ProxBinarsity(ProxWithGroups):
    """Proximal operator of binarsity. It is simply a succession of two steps on
    different intervals: ``ProxTV`` plus a centering translation. More
    precisely, total-variation regularization is applied on a coefficient vector
    being a concatenation of multiple coefficient vectors corresponding to
    blocks, followed by centering within sub-blocks. Blocks (non-overlapping)
    are specified by the ``blocks_start`` and ``blocks_length`` parameters.

    Parameters
    ----------
    strength : `float`
        Level of total-variation penalization

    blocks_start : `np.array`, shape=(n_blocks,)
        First entry of each block

    blocks_length : `np.array`, shape=(n_blocks,)
        Size of each block

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply in the end a projection onto the set of vectors with
        non-negative entries

    Attributes
    ----------
    n_blocks : `int`
        Number of blocks

    dtype : `{'float64', 'float32'}`
        Type of the arrays used.

    References
    ----------
    ProxBinarsity uses the fast-TV algorithm described in:

    Condat, L. (2012).
    `A Direct Algorithm for 1D Total Variation Denoising`_.

    .. _A Direct Algorithm for 1D Total Variation Denoising: https://hal.archives-ouvertes.fr/hal-00675043v2/document
    """

    def __init__(self, strength: float, blocks_start, blocks_length,
                 range: tuple = None, positive: bool = False):
        ProxWithGroups.__init__(self, strength, blocks_start, blocks_length,
                                range, positive)
        self._prox = self._build_cpp_prox("float64")

    def _get_dtype_map(self):
        return dtype_map
