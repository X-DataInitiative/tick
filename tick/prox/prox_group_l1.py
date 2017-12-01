# License: BSD 3 clause


from .base import ProxWithGroups
from .build.prox import ProxGroupL1 as _ProxGroupL1


class ProxGroupL1(ProxWithGroups):
    """Proximal operator of group-L1, a.k.a group-Lasso. It applies `ProxL2` in
    each group, or block. Blocks (non-overlapping) are specified by the
    ``blocks_start`` and ``blocks_length`` parameters.

    Parameters
    ----------
    strength : `float`
        Level of penalization. Note that `ProxL2` which is applied in each group
        multiplies ``strength`` by the square-root of the group size.
        This allows to consider strengths that have the same order as with
        `ProxL1` or other separable proximal operators.

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
    def __init__(self, strength: float, blocks_start, blocks_length,
                 range: tuple = None, positive: bool = False):
        ProxWithGroups.__init__(self, strength, blocks_start, blocks_length,
                                range, positive)

    def _get_prox_class(self):
        return _ProxGroupL1
