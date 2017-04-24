import warnings
from itertools import product

import numpy as np
from tick.simulation.base import SimuPointProcess
from numpy.linalg import eig, inv

from .build.simulation import Hawkes as _Hawkes
from .hawkes_kernels import HawkesKernel0


class SimuHawkes(SimuPointProcess):
    """Hawkes process simulation
    
    They are defined by the intensity:
    
    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^D \\int \\phi_{ij}(t - s) dN_j(s)

    where
    
    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels
    * :math:`dN_j` are the processes differentiates

    Parameters
    ----------
    kernels : `np.ndarray`, shape=(n_nodes, n_nodes)
        A 2-dimensional arrays of kernels, also noted :math:`\phi_{ij}`

    baseline : `np.ndarray`, shape=(n_nodes, )
        The baseline of all intensities, also noted :math:`\mu`

    n_nodes : `int`
        The number of nodes of the Hawkes process. If kernels and baseline
        are None this will create a Hawkes process with n_nodes nodes of zero
        kernels and with zero baseline

    end_time : `float`, default=None
        Time until which this point process will be simulated

    max_jumps : `int`, default=None
        Simulation will stop if this number of jumps in reached

    seed : `int`, default = None
        The seed of the random sampling. If it is None then a random seed
        (different at each run) will be chosen.

    force_simulation : `bool`, default = False
        If force is not set to True, simulation won't be run if the matrix of
        the L1 norm of kernels has a spectral radius greater or equal to 1 as
        it would be unstable

    Attributes
    ----------
    timestamps : `list` of `np.ndarray`, size=n_nodes
        A list of n_nodes timestamps arrays, each array containing the
        timestamps of all the jumps for this node

    simulation_time : `float`
        Time until which this point process has been simulated

    n_total_jumps : `int`
        Total number of jumps simulated

    tracked_intensity : `list[np.ndarray]`, size=n_nodes
        A record of the intensity with which this point process has been
        simulated.
        Note: you must call track_intensity before simulation to record it

    intensity_tracked_times : `np.ndarray`
        The times at which intensity has been recorded.
        Note: you must call track_intensity before simulation to record it

    intensity_track_step : `float`
        Step with which the intensity has been recorded
    """

    _attrinfos = {
        "kernels": {"writable": False},
        "_kernel_0": {"writable": False},
    }

    def __init__(self, kernels=None, baseline=None, n_nodes=None,
                 end_time=None, max_jumps=None, seed=None, verbose=True,
                 force_simulation=False):
        SimuPointProcess.__init__(self, end_time=end_time, max_jumps=max_jumps,
                                  seed=seed, verbose=verbose)

        self.force_simulation = force_simulation
        # We keep a reference on this kernel to avoid copies
        self._kernel_0 = HawkesKernel0()

        if isinstance(kernels, list):
            kernels = np.array(kernels)

        if isinstance(baseline, list):
            baseline = np.array(baseline)

        if baseline is not None and baseline.dtype != float:
            baseline = baseline.astype(float)

        self.check_parameters_coherence(kernels, baseline, n_nodes)

        # Init _pp so we hae access to self.n_nodes
        if n_nodes is None:
            if baseline is not None:
                n_nodes = baseline.shape[0]
            else:
                n_nodes = kernels.shape[0]

        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive but equals %i" % n_nodes)
        self._pp = _Hawkes(n_nodes, self._pp_init_seed)

        if kernels is not None:
            if kernels.shape != (self.n_nodes, self.n_nodes):
                raise ValueError("kernels shape should be %s instead of %s" %
                                 ((self.n_nodes, self.n_nodes),
                                  kernels.shape))
            self.kernels = kernels
            self._init_kernels()
        else:
            self._init_zero_kernels()

        if baseline is not None:
            if baseline.shape != (self.n_nodes,):
                raise ValueError("baseline shape should be %s instead of %s" %
                                 ((self.n_nodes,), self.baseline.shape))
            self.baseline = baseline
            self._init_baseline()

        else:
            self._init_zero_baseline()

    def _init_zero_kernels(self):
        self.kernels = np.empty((self.n_nodes, self.n_nodes), dtype=object)
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            self.kernels[i, j] = self._kernel_0

    def _init_zero_baseline(self):
        self.baseline = np.empty((self.n_nodes,), dtype=object)
        for i in range(self.n_nodes):
            self.set_baseline(i, 0)

    def _init_baseline(self):
        for i in range(self.n_nodes):
            self.set_baseline(i, self.baseline[i])

    def _init_kernels(self):
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            if self.kernels[i, j] == 0:
                kernel_ij = self._kernel_0
            else:
                kernel_ij = self.kernels[i, j]
            self.set_kernel(i, j, kernel_ij)

    def check_parameters_coherence(self, kernels, baseline, n_nodes):
        set_kernels = kernels is not None
        set_baseline = baseline is not None
        set_n_nodes = n_nodes is not None

        if set_n_nodes and (set_kernels or set_baseline):
            raise ValueError("n_nodes will be automatically calculated if "
                             "baseline or kernels is set")

        if not set_n_nodes and not set_kernels and not set_baseline:
            raise ValueError("n_nodes must be given if neither kernels, "
                             "nor baseline are given")

        if set_kernels and set_baseline and len(kernels) != len(baseline):
            raise ValueError("kernels and baseline have different length. "
                             "kernels has length %i, whereas baseline has "
                             "length %i." % (len(kernels), len(baseline)))

    def set_kernel(self, i, j, kernel):
        if isinstance(kernel, (int, float)) and kernel == 0:
            self.kernels[i, j] = self._kernel_0
            self._pp.set_kernel(i, j, self._kernel_0._kernel)
        else:
            self.kernels[i, j] = kernel
            self._pp.set_kernel(i, j, kernel._kernel)

    def set_baseline(self, i, baseline):
        self.baseline[i] = baseline
        self._pp.set_mu(i, baseline)

    def _simulate(self):
        """Launch simulation of the Hawkes process by thinning
        """
        if np.linalg.norm(self.baseline) == 0:
            warnings.warn("Baselines have not been set, hence this hawkes "
                          "process won't jump")

        if not self.force_simulation and self.spectral_radius() >= 1 \
                and self.max_jumps is None:
            raise ValueError("Simulation not launched as this Hawkes process "
                             "is not stable (spectral radius of %.2g). You "
                             "can use force_simulation parameter if you "
                             "really want to simulate it"
                             % self.spectral_radius())

        SimuPointProcess._simulate(self)

    def spectral_radius(self):
        """Compute the spectral radius of the matrix of l1 norm of Hawkes
        kernels.

        Notes
        -----
        If the spectral radius is greater that 1, the hawkes process is not
        stable
        """

        get_norm = np.vectorize(lambda kernel: kernel.get_norm())
        norms = get_norm(self.kernels)

        # It might happens that eig returns a complex number but with a
        # negligible complex part, in this case we keep only the real part
        spectral_radius = max(eig(norms)[0])
        spectral_radius = np.real_if_close(spectral_radius)
        return spectral_radius

    def mean_intensity(self):
        """Compute the mean intensity vector
        """
        get_norm = np.vectorize(lambda kernel: kernel.get_norm())
        norms = get_norm(self.kernels)
        return inv(np.eye(self.n_nodes) - norms).dot(self.baseline)
