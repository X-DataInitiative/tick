# License: BSD 3 clause

from tick.hawkes.simulation.build.hawkes_simulation import (HawkesKernelExp as
                                                            _HawkesKernelExp)
from .hawkes_kernel import HawkesKernel


class HawkesKernelExp(HawkesKernel):
    """Hawkes kernel with exponential decay

    .. math::
        \\phi(t) = \\alpha \\beta \\exp (- \\beta t) 1_{t > 0}

    where :math:`\\alpha` is the intensity of the kernel and
    :math:`\\beta` its decay.

    Parameters
    ----------
    intensity : `float`
        Intensity of the kernel, also noted :math:`\\alpha`

    decay : `float`
        Decay of the kernel, also noted :math:`\\beta`
    """

    def __init__(self, intensity, decay):
        HawkesKernel.__init__(self)
        self._kernel = _HawkesKernelExp(intensity, decay)

    @property
    def intensity(self):
        return self._kernel.get_intensity()

    @property
    def decay(self):
        return self._kernel.get_decay()

    def __str__(self):
        if self.intensity == 0:
            return "0"
        elif self.decay == 0:
            return "{:g}".format(self.intensity)
        else:
            return "{:g} * {:g} * exp(- {:g} * t)".format(
                self.intensity, self.decay, self.decay)

    def __repr__(self):
        return self.__str__().replace(" ", "")

    def __strtex__(self):
        if self.intensity == 0:
            return r"$0$"
        elif self.decay == 0:
            return r"${:g}$".format(self.intensity)
        else:
            if self.intensity * self.decay == 1:
                if self.decay == 1:
                    return r"$e^{-t}$"
                else:
                    return r"$e^{-%g t}$" % self.decay
            else:
                if self.decay == 1:
                    return r"$%g e^{- t}$" % (self.intensity * self.decay)
                else:
                    return r"$%g e^{-%g t}$" % (self.intensity * self.decay,
                                                self.decay)
