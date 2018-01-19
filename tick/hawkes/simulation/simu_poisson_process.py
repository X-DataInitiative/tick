# License: BSD 3 clause

import numpy as np

from tick.hawkes.simulation.base import SimuPointProcess
from tick.hawkes.simulation.build.hawkes_simulation import Poisson as _Poisson


class SimuPoissonProcess(SimuPointProcess):
    """Homogeneous Poisson process simulation

    Parameters
    ----------
    intensities : `float` or a `np.ndarray`
        The intensities of the poisson process. If float this Poisson process
        has one node, otherwise it is multidimensional

    end_time : `float`, default=None
        Time until which this point process will be simulated

    max_jumps : `int`, default=None
        Simulation will stop if this number of jumps in reached

    seed : `int`, default = None
        The seed of the random sampling. If it is None then a random seed
        (different at each run) will be chosen.

    verbose : `bool`, default=True
        If True, simulation information is printed

    Attributes
    ----------
    n_nodes : `int`
        The number of nodes of the point process

    end_time : `float`
        Time until which this point process has been simulated

    n_total_jumps : `int`
        Total number of jumps simulated

    timestamps : `list` of `np.ndarray`, size=n_nodes
        A list of n_nodes timestamps arrays, each array containing the
        timestamps of all the jumps for this node

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

    def __init__(self, intensities, end_time=None, max_jumps=None,
                 verbose=True, seed=None):
        SimuPointProcess.__init__(self, end_time=end_time, max_jumps=max_jumps,
                                  seed=seed, verbose=verbose)

        if intensities.__class__ == list:
            intensities = np.array(intensities, dtype=float)
        if intensities.__class__ == np.ndarray and intensities.dtype != float:
            intensities = intensities.astype(float)

        self._pp = _Poisson(intensities, self._pp_init_seed)

    @property
    def intensities(self):
        return self._pp.get_intensities()
