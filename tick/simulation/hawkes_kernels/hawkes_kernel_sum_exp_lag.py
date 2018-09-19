# License: BSD 3 clause

import numpy as np

from . import HawkesKernelExp
from .hawkes_kernel import HawkesKernel
from ..build.simulation import HawkesKernelSumExpLag as _HawkesKernelSumExpLag


class HawkesKernelSumExpLag(HawkesKernel):


    def __init__(self, intensities, decays, lags):
        HawkesKernel.__init__(self)

        if intensities.__class__ == list:
            intensities = np.array(intensities, dtype=float)
        if intensities.dtype != float:
            intensities = intensities.astype(float)
        if decays.__class__ == list:
            decays = np.array(decays, dtype=float)
        if decays.dtype != float:
            decays = decays.astype(float)

        self._kernel = _HawkesKernelSumExpLag(intensities, decays, lags)

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
        return [HawkesKernelExp(intensity, decay)
                for (intensity, decay) in zip(self.intensities, self.decays)]

    def __str__(self):
        return " + ".join([str(kernel) for kernel in
                           self._generate_corresponding_single_exp_kernels()])

    def __repr__(self):
        return " + ".join([kernel.__repr__() for kernel in
                           self._generate_corresponding_single_exp_kernels()])

    def __strtex__(self):
        return " + ".join([kernel.__strtex__() for kernel in
                           self._generate_corresponding_single_exp_kernels()])
