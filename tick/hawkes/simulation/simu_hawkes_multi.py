# License: BSD 3 clause

import copy
import multiprocessing
from multiprocessing import Pool

import numpy as np

from tick.base.simulation import Simu


def simulate_single(simulation):
    simulation.simulate()
    return simulation


class SimuHawkesMulti(Simu):
    """Parallel simulations of a single Hawkes simulation

    The incoming Hawkes simulation is replicated by the number n_simulations. At
    simulation time, the replicated Hawkes processes are run in parallel on a
    number of threads specified by n_threads.

    Attributes
    ----------
    hawkes_simu : 'SimuHawkes'
        The Hawkes simulation that is replicated and simulated in parallel

    n_simulations : `int`
        The number of times the Hawkes simulation is performed

    n_threads : `int`, default=1
        The number of threads used to run the Hawkes simulations. If this number
        is negative or zero, the number of threads is set to the number of
        system available CPU cores.

    n_total_jumps : `list` of `int`
        List of the total number of jumps simulated for each process

    timestamps : `list` of `list` of `np.ndarray`, size=n_simulations, n_nodes
        A list containing n_simulations lists of timestamps arrays, one for each
        process that is being simulated by this object.

    end_time : `list` of `float`
        List of the end time for each Hawkes process

    max_jumps : `list` of `int`
        List of the maximum number of jumps for each process

    simulation_time : `list` of `float`
        List of times each process has been simulated

    n_nodes : `list` of `int`
        List of the number of nodes of the Hawkes processes

    spectral_radius : `list` of `float`
        List of the spectral radii of the Hawkes processes

    mean_intensity : `list` of `float`
        List of the mean intensities of the Hawkes processes

    """

    _attrinfos = {
        "_simulations": {},
        "hawkes_simu": {
            "writable": False
        },
        "n_simulations": {
            "writable": False
        },
    }

    def __init__(self, hawkes_simu, n_simulations, n_threads=1):
        self.hawkes_simu = hawkes_simu
        self.n_simulations = n_simulations

        if n_threads <= 0:
            n_threads = multiprocessing.cpu_count()

        self.n_threads = n_threads

        if n_simulations <= 0:
            raise ValueError("n_simulations must be greater or equal to 1")

        self._simulations = [
            copy.deepcopy(hawkes_simu) for _ in range(n_simulations)
        ]

        Simu.__init__(self, seed=self.seed, verbose=hawkes_simu.verbose)

        if self.seed is not None and self.seed >= 0:
            self.reseed_simulations(self.seed)

    @property
    def seed(self):
        return self.hawkes_simu.seed

    @seed.setter
    def seed(self, val):
        self.reseed_simulations(val)

    def reseed_simulations(self, seed):
        """Reseeds all simulations such that each simulation is started with a
        unique seed. The random selection of new seeds is seeded with the value
        given in 'seed'.

        Parameters
        ----------
        seed :
            Seed used to randomly select new seeds
        """
        # this updates self.seed
        self.hawkes_simu._pp.reseed_random_generator(seed)

        if seed >= 0:
            np.random.seed(self.seed)
            new_seeds = np.random.randint(0, 2 ** 31 - 1, self.n_simulations)
            new_seeds = new_seeds.astype('int32')

        else:
            new_seeds = np.ones(self.n_simulations, dtype='int32') * seed

        for simu, seed in zip(self._simulations, new_seeds):
            simu.seed = seed.item()

    @property
    def n_total_jumps(self):
        return [simu.n_total_jumps for simu in self._simulations]

    @property
    def timestamps(self):
        return [simu.timestamps for simu in self._simulations]

    @property
    def end_time(self):
        return [simu.end_time for simu in self._simulations]

    @end_time.setter
    def end_time(self, end_times):
        if len(end_times) != self.n_simulations:
            raise ValueError('end_time must have length {}'.format(
                self.n_simulations))
        for i, simu in enumerate(self._simulations):
            simu.end_time = end_times[i]

    @property
    def max_jumps(self):
        return [simu.max_jumps for simu in self._simulations]

    @property
    def simulation_time(self):
        return [simu.simulation_time for simu in self._simulations]

    @property
    def n_nodes(self):
        return [simu.n_nodes for simu in self._simulations]

    @property
    def spectral_radius(self):
        return [simu.spectral_radius() for simu in self._simulations]

    @property
    def mean_intensity(self):
        return [simu.mean_intensity() for simu in self._simulations]

    def get_single_simulation(self, i):
        return self._simulations[i]

    def _simulate(self):
        """ Launches a series of n_simulations Hawkes simulation in a thread
        pool
        """
        with Pool(self.n_threads) as p:
            self._simulations = p.map(simulate_single, self._simulations)
