# License: BSD 3 clause

from tick.hawkes.simulation.build.hawkes_simulation import (
    HawkesKernelPowerLaw as _HawkesKernelPowerLaw)
from .hawkes_kernel import HawkesKernel


class HawkesKernelPowerLaw(HawkesKernel):
    """Hawkes kernel for power law

    .. math::
        \\phi(t) = \\phi(t) = \\alpha (\\delta + t)^{- \\beta} 1_{t > 0}

    Where :math:`\\alpha` is called the multiplier, `\\delta` the cut-off and
    :math:`\\beta` the exponent

    Parameters
    ----------
    multiplier : `float`
        Multiplier of the kernel, also noted :math:`\\alpha`

    cutoff : `float`
        Cut-off of the kernel, also noted :math:`\\delta`

    exponent : `float`
        Exponent of the kernel, also noted :math:`\\beta`
    """

    def __init__(self, multiplier, cutoff, exponent, support=-1, error=1e-5):
        HawkesKernel.__init__(self)
        self._kernel = _HawkesKernelPowerLaw(multiplier, cutoff, exponent,
                                             support, error)

    @property
    def multiplier(self):
        return self._kernel.get_multiplier()

    @property
    def cutoff(self):
        return self._kernel.get_cutoff()

    @property
    def exponent(self):
        return self._kernel.get_exponent()

    def __str__(self):
        if self.multiplier == 0:
            return '0'
        elif self.exponent == 0:
            return '{:g}'.format(self.multiplier)
        else:
            return '{:g} * ({:g} + t)^(-{:g})'.format(
                self.multiplier, self.cutoff, self.exponent)

    def __repr__(self):
        return self.__str__().replace(' ', '')

    def __strtex__(self):
        if self.multiplier == 0:
            return r'$0$'
        elif self.exponent == 0:
            return r'${:g}$'.format(self.multiplier)
        else:
            if self.multiplier == 1:
                return r'$(%g+t)^{-%g}$' % (self.cutoff, self.exponent)
            else:
                return r'$%g (%g+t)^{-%g}$' % (self.multiplier, self.cutoff,
                                               self.exponent)
