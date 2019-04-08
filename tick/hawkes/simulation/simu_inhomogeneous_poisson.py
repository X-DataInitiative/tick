# License: BSD 3 clause

import numpy as np

from tick.base import TimeFunction
from tick.hawkes.simulation.base import SimuPointProcess
from tick.hawkes.simulation.build.hawkes_simulation import (
    InhomogeneousPoisson as _InhomogeneousPoisson)


class SimuInhomogeneousPoisson(SimuPointProcess):
    """Inhomogeneous Poisson process simulation

    Parameters
    ----------
    intensities_functions : `list`of `TimeFunction`
        The intensities functions of the inhomogeneous Poisson process

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

    def __init__(self, intensities_functions, end_time=None, max_jumps=None,
                 seed=None, verbose=True):
        SimuPointProcess.__init__(self, end_time=end_time, max_jumps=max_jumps,
                                  seed=seed, verbose=verbose)
        cpp_obj_list = [
            intensity_function._time_function
            for intensity_function in intensities_functions
        ]
        self._pp = _InhomogeneousPoisson(cpp_obj_list, self._pp_init_seed)

    def intensity_value(self, node, times):
        return self._pp.intensity_value(node, times)
