# License: BSD 3 clause

import numpy as np

from tick.hawkes.simulation.build.hawkes_simulation import (
    HawkesKernelSumExp as _HawkesKernelSumExp)
from . import HawkesKernelExp
from .hawkes_kernel import HawkesKernel


class HawkesKernelSumExp(HawkesKernel):
    """Hawkes kernel with sum exponential decays

    .. math::
        \\phi(t) = \\sum_{u=1}^{U} \\alpha_u \\beta_u \\exp (- \\beta_u t) 
                                   1_{t > 0}

    where :math:`\\alpha_u` are the intensity of the kernel
    and :math:`\\beta_u` its decays.

    Parameters
    ----------
    intensities : `np.ndarray`, shape = (n_decays, )
        Intensity of the kernel, also noted :math:`\\alpha`

    decays : `np.ndarray`, shape = (n_decays, )
        Decay of the kernel, also noted :math:`\\beta`

    Attributes
    ----------
    n_decays : `int`
        Number of decays of the kernel, also noted :math:`U`
    """

    def __init__(self, intensities, decays):
        HawkesKernel.__init__(self)

        if intensities.__class__ == list:
            intensities = np.array(intensities, dtype=float)
        if intensities.dtype != float:
            intensities = intensities.astype(float)
        if decays.__class__ == list:
            decays = np.array(decays, dtype=float)
        if decays.dtype != float:
            decays = decays.astype(float)

        self._kernel = _HawkesKernelSumExp(intensities, decays)

    @property
    def intensities(self):
        return self._kernel.get_intensities()

    @property
    def decays(self):
        return self._kernel.get_decays()

    @property
    def n_decays(self):
        return self._kernel.get_n_decays()

    def _generate_corresponding_single_exp_kernels(self):
        return [
            HawkesKernelExp(intensity, decay)
            for (intensity, decay) in zip(self.intensities, self.decays)
        ]

    def __str__(self):
        return " + ".join([
            str(kernel)
            for kernel in self._generate_corresponding_single_exp_kernels()
        ])

    def __repr__(self):
        return " + ".join([
            kernel.__repr__()
            for kernel in self._generate_corresponding_single_exp_kernels()
        ])

    def __strtex__(self):
        return " + ".join([
            kernel.__strtex__()
            for kernel in self._generate_corresponding_single_exp_kernels()
        ])
