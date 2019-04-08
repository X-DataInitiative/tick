# License: BSD 3 clause

from abc import ABC, abstractmethod
from time import time
import numpy as np

from tick.base import Base


class Simu(ABC, Base):
    """
    Abstract simulation class. It does nothing besides printing and
    verbosing.

    Parameters
    ----------
    seed : `int`
        The seed of the random number generator

    verbose : `bool`
        If True, print things

    Attributes
    ----------
    time_start : `str`
        Start date of the simulation

    time_elapsed : `int`
        Duration of the simulation, in seconds

    time_end : `str`
        End date of the simulation
    """

    _attrinfos = {
        "time_start": {
            "writable": False
        },
        "time_elapsed": {
            "writable": False
        },
        "time_end": {
            "writable": False
        },
        "_time_start": {
            "writable": False
        }
    }

    def __init__(self, seed: int = None, verbose: bool = True):
        Base.__init__(self)
        self.seed = seed
        self.verbose = verbose
        if seed is not None and seed >= 0:
            self._set_seed()
        self._set("time_start", None)
        self._set("time_elapsed", None)
        self._set("time_end", None)
        self._set("_time_start", None)

    def _set_seed(self):
        np.random.seed(self.seed)

    def _start_simulation(self):
        self._set("time_start", self._get_now())
        self._set("_time_start", time())
        if self.verbose:
            msg = "Launching simulation using {class_}..." \
                    .format(class_=self.name)
            print("-" * len(msg))
            print(msg)

    def _end_simulation(self):
        self._set("time_end", self._get_now())
        t = time()
        self._set("time_elapsed", t - self._time_start)
        if self.verbose:
            msg = "Done simulating using {class_} in {time:.2e} " \
                  "seconds." \
                .format(class_=self.name, time=self.time_elapsed)
            print(msg)

    @abstractmethod
    def _simulate(self):
        pass

    def simulate(self):
        """Launch the simulation of data
        """
        self._start_simulation()
        result = self._simulate()
        self._end_simulation()
        return result

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("coeffs", None)
        return dd
