# License: BSD 3 clause

from itertools import product

import numpy as np

from tick.hawkes.simulation import SimuHawkes
from .hawkes_kernels import HawkesKernelExp, HawkesKernel0


class SimuHawkesExpKernels(SimuHawkes):
    """Hawkes process with exponential kernels simulation
    
    They are defined by the intensity:
    
    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^D \\int \\phi_{ij}(t - s) dN_j(s)
    
    where
    
      - :math:`D` is the number of nodes
      - :math:`\mu_i(t)` are the baseline intensities
      - :math:`\phi_{ij}` are the kernels
      - :math:`dN_j` are the processes differentiates

    and with an exponential parametrisation of the kernels

    .. math::
        \\phi_{ij}(t) = \\alpha_{ij} \\beta_{ij} \\exp (- \\beta_{ij} t) 1_{t > 0}

    where :math:`\\alpha_{ij}` is the intensity of the kernel and
    :math:`\\beta_{ij}` its decay. The matrix of all :math:`\\alpha` is called
    adjacency matrix.

    Parameters
    ----------
    baseline : `np.ndarray` or `list`
        The baseline of all intensities, also noted :math:`\mu(t)`. It might 
        be three different types:
        
        * `np.ndarray`, shape=(n_nodes,) : One baseline per node is given. 
          Hence baseline is assumed to be constant, ie. 
          :math:`\mu_i(t) = \mu_i`
        * `np.ndarray`, shape=(n_nodes, n_intervals) : `n_intervals` baselines 
          are given per node. This assumes parameter `period_length` is also 
          given. In this case baseline is piecewise constant on intervals of 
          size `period_length / n_intervals` and periodic.
        * `list` of `tick.base.TimeFunction`, shape=(n_nodes,) : One function 
          is given per node, ie. :math:`\mu_i(t)` is explicitely given.

    adjacency : `np.ndarray`, shape=(n_nodes, n_nodes)
        Intensities of exponential kernels, also named :math:`\\alpha_{ij}`

    decays : `float` or `np.ndarray`, shape=(n_nodes, n_nodes)
        Decays of exponential kernels, also named :math:`\\beta_{ij}`
        If a `float` is given, all decays are equal to this float

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
        "adjacency": {
            "writable": False
        },
        "decays": {
            "writable": False
        },
    }

    def __init__(self, adjacency, decays, baseline=None, end_time=None,
                 period_length=None, max_jumps=None, seed=None, verbose=True,
                 force_simulation=False):

        if isinstance(adjacency, list):
            adjacency = np.array(adjacency)

        if isinstance(decays, list):
            decays = np.array(decays)

        n_nodes = adjacency.shape[0]

        if adjacency.shape != (n_nodes, n_nodes):
            raise ValueError("adjacency matrix should be squared and its "
                             "shape is %s" % str(adjacency.shape))

        decays_is_number = isinstance(decays, (int, float))
        if not decays_is_number and decays.shape != (n_nodes, n_nodes):
            raise ValueError("decays should be either a float or an array of "
                             "shape %s but its shape is %s" % (str(
                                 (n_nodes, n_nodes)), str(decays.shape)))

        self.adjacency = adjacency
        self.decays = decays

        kernels = self._build_exp_kernels()

        SimuHawkes.__init__(self, kernels=kernels, baseline=baseline,
                            end_time=end_time, period_length=period_length,
                            max_jumps=max_jumps, seed=seed, verbose=verbose,
                            force_simulation=force_simulation)

    def _build_exp_kernels(self):
        """Build exponential kernels from adjacency and decays
        """
        decays_is_number = isinstance(self.decays, (int, float))
        n_nodes = self.adjacency.shape[0]

        kernel_0 = HawkesKernel0()
        kernels = np.empty_like(self.adjacency, dtype=object)
        for i, j in product(range(n_nodes), range(n_nodes)):
            if decays_is_number:
                kernel_decay = self.decays
            else:
                kernel_decay = self.decays[i, j]
            kernel_intensity = self.adjacency[i, j]
            if kernel_intensity == 0:
                kernels[i, j] = kernel_0
            else:
                kernels[i, j] = HawkesKernelExp(kernel_intensity, kernel_decay)

        return kernels

    def adjust_spectral_radius(self, spectral_radius):
        """Adjust the spectral radius of the matrix of l1 norm of Hawkes
        kernels.

        Parameters
        ----------
        spectral_radius : `float`
            The targeted spectral radius
        """

        original_spectral_radius = self.spectral_radius()

        adjacency = self.adjacency * \
                    spectral_radius / original_spectral_radius

        self._set("adjacency", adjacency)

        kernels = self._build_exp_kernels()
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            self.set_kernel(i, j, kernels[i, j])
