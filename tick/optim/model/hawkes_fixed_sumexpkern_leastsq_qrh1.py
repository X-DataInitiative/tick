# License: BSD 3 clause

import numpy as np

from .base import ModelFirstOrder

from tick.optim.model.build.model import ModelHawkesFixedSumExpKernLeastSqQRH1 as _ModelHawkesFixedSumExpKernLeastSqQRH1

class ModelHawkesFixedSumExpKernLeastSqQRH1(ModelFirstOrder):

    _attrinfos = {
        "decays": {
            "cpp_setter": "set_decays"
        },
        "_end_times": {},
        "data": {}
    }


    def __init__(self, decays: np.ndarray, MaxN : int, n_threads: int = 1, optimization_level: int = 0):
        ModelFirstOrder.__init__(self)
        self._model = _ModelHawkesFixedSumExpKernLeastSqQRH1(decays, MaxN, n_threads, optimization_level)
        print(self._model)


    def fit(self, events, global_n, end_times=None):
        """Set the corresponding realization(s) of the process.

        Parameters
        ----------
        events : `list` of `list` of `np.ndarray`
            List of Hawkes processes realizations.
            Each realization of the Hawkes process is a list of n_node for
            each component of the Hawkes. Namely `events[i][j]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component j of realization i.
            If only one realization is given, it will be wrapped into a list

        end_times : `np.ndarray` or `float`, default = None
            List of end time of all hawkes processes that will be given to the
            model. If None, it will be set to each realization's latest time.
            If only one realization is provided, then a float can be given.
        """
        self._end_times = end_times
        return ModelFirstOrder.fit(self, events, global_n, end_times)


    def _set_data(self, events, global_n, end_times):
        """Set the corresponding realization(s) of the process.

        Parameters
        ----------
        events : `list` of `list` of `np.ndarray`
            List of Hawkes processes realizations.
            Each realization of the Hawkes process is a list of n_node for
            each component of the Hawkes. Namely `events[i][j]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component j of realization i.
            If only one realization is given, it will be wrapped into a list
        """

        # end_times = self._end_times
        if end_times is None:
            end_times = max(map(max, events))

        self._model.set_data(events, global_n, end_times)


    def _loss(self, coeffs):
        return self._model.loss(coeffs)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._model.grad(coeffs, out)
        return out

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()

    @property
    def decays(self):
        return self._model.get_decays()

    @property
    def n_jumps(self):
        return self._model.get_n_total_jumps()
