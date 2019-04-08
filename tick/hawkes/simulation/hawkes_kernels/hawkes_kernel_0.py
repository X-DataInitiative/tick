# License: BSD 3 clause

from tick.hawkes.simulation.build.hawkes_simulation import (HawkesKernel0 as
                                                            _HawkesKernel0)
from .hawkes_kernel import HawkesKernel


class HawkesKernel0(HawkesKernel):
    """Hawkes zero kernel

    .. math::
        \phi(t) = 0
    """

    def __init__(self):
        HawkesKernel.__init__(self)
        self._kernel = _HawkesKernel0()

    def __str__(self):
        return '0'

    def __repr__(self):
        return '0'

    def __strtex__(self):
        return r'$0$'
