# License: BSD 3 clause

import warnings

from tick.base.simulation import Simu


class SimuPointProcess(Simu):
    """Base class for point process simulation

    This class is not meant to be used directly

    Parameters
    ----------
    seed : `int`, default = None
        The seed of the random sampling. If it is None then a random seed
        (different at each run) will be chosen.

    verbose : `bool`, default=True
        If True, information is printed

    end_time : `float`, default=None
        Time until which this point process will be simulated

    max_jumps : `int`, default=None
        Simulation will stop if this number of jumps in reached

    Attributes
    ----------
    n_nodes : `int`
        The number of nodes of the point process

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

    simulation_time : `float`, default=None
        Time until which this point process has been simulated
    """

    _attrinfos = {
        "_pp": {
            "writable": False
        },
        "_pp_init_seed": {
            "writable": True
        },
        "_end_time": {
            "writable": False
        },
    }

    def __init__(self, end_time=None, max_jumps=None, seed=None, verbose=True):
        self._pp = None
        Simu.__init__(self, seed=seed, verbose=verbose)

        self._end_time = None
        self.end_time = end_time
        self.max_jumps = max_jumps

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, val):
        if val is not None and val < self.simulation_time:
            raise ValueError("This point process has already been simulated "
                             "until time %f, you cannot set a smaller "
                             "end_time (%f)" % (self.simulation_time, val))
        self._set('_end_time', val)

    @property
    def n_nodes(self):
        return self._pp.get_n_nodes()

    @property
    def simulation_time(self):
        if self._pp is None:
            return 0
        return self._pp.get_time()

    @property
    def n_total_jumps(self):
        return self._pp.get_n_total_jumps()

    @property
    def timestamps(self):
        return self._pp.get_timestamps()

    @property
    def tracked_intensity(self):
        if not self.is_intensity_tracked():
            raise ValueError("Intensity has not been tracked, you should call "
                             "track_intensity before simulation")
        return self._pp.get_itr()

    @property
    def intensity_tracked_times(self):
        if not self.is_intensity_tracked():
            raise ValueError("Intensity has not been tracked, you should call "
                             "track_intensity before simulation")
        return self._pp.get_itr_times()

    @property
    def intensity_track_step(self):
        return self._pp.get_itr_step()

    @property
    def seed(self):
        if self._pp is None:
            # _pp_init_seed is only used to create point process object
            return self._pp_init_seed
        else:
            return self._pp.get_seed()

    @seed.setter
    def seed(self, val):
        if val is None:
            val = -1
        # _pp_init_seed is only used to create point process object
        self._pp_init_seed = val
        if self._pp is not None:
            self._pp.reseed_random_generator(val)

    def _simulate(self):
        """Launch simulation of the process by thinning

        Parameters
        ----------
        end_time : `float`
            The time until which the process will be simulated

        n_points : `int`
            The number of points until we keep simulating. Beware this
            introduces a small bias especially if the number of points is small
        """
        if self.end_time == self.simulation_time:
            warnings.warn("This process has already be simulated until time %f"
                          % self.end_time)

        if self.end_time is None and self.max_jumps is None:
            raise (ValueError('Either end_time or max_jumps must be set'))

        elif self.end_time is not None and self.max_jumps is None:
            self._pp.simulate(float(self.end_time))

        elif self.end_time is None and self.max_jumps is not None:
            self._pp.simulate(int(self.max_jumps))

        elif self.end_time is not None and self.max_jumps is not None:
            self._pp.simulate(self.end_time, self.max_jumps)

    def track_intensity(self, intensity_track_step=-1):
        """Activate the tracking of the intensity

        Parameters
        ----------
        intensity_track_step : `float`
            If positive then the step the intensity vector is recorded every,
            otherwise, it is deactivated.

        Notes
        -----
        This method must be called before simulation
        """
        self._pp.activate_itr(intensity_track_step)

    def is_intensity_tracked(self):
        """Is intensity tracked thanks to track_intensity or not
        """
        return self._pp.itr_on()

    def reset(self):
        """Reset the process, so that is is ready for a brand new simulation
        """
        self._pp.reset()

    def threshold_negative_intensity(self, allow=True):
        """Threshold intensity to 0 if it becomes negative.
        This allows simulation with negative kernels

        Parameters
        ----------
        allow : `bool`
            Flag to allow negative intensity thresholding
        """
        self._pp.set_threshold_negative_intensity(allow)

    def set_timestamps(self, timestamps, end_time=None):
        if end_time is None:
            end_time = max(map(max, timestamps))

        self.end_time = end_time
        self._pp.set_timestamps(timestamps, end_time)
