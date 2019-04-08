# License: BSD 3 clause

import numpy as np
from scipy.sparse import sputils, csr_matrix

from tick.base_model import N_CALLS_LOSS, PASS_OVER_DATA
from tick.base_model.model_first_order import ModelFirstOrder
from tick.hawkes.model.build.hawkes_model import (
    ModelHawkesSumExpKernLeastSq,
    ModelHawkesExpKernLeastSq,
    ModelHawkesSumExpKernLogLik,
)


class ModelHawkes(ModelFirstOrder):
    """Base class of Hawkes models

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    _attrinfos = {
        "approx": {
            "writable": False
        },
        "data": {
            "writable": False
        },
        "_end_times": {
            "writable": False
        },
        "n_threads": {
            "writable": True,
            "cpp_setter": "set_n_threads"
        },
    }

    def __init__(self, approx: int = 0, n_threads: int = 1):
        ModelFirstOrder.__init__(self)

        self.approx = approx
        self.n_threads = n_threads
        self.data = None
        self._end_times = None
        self._model = None
        self.dtype = np.dtype("float64")

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()

    def fit(self, data, end_times=None):
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
        self._set('_end_times', end_times)
        return ModelFirstOrder.fit(self, data)

    def _set_data(self, events):
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
        self._set("data", events)
        if len(events[0]) == 0 or not isinstance(events[0][0], np.ndarray):
            events = [events]

        end_times = self._end_times
        if end_times is None:
            non_empty_events = [[r for r in e if len(r) > 0] for e in events]
            end_times = np.array([max(map(max, e)) for e in non_empty_events])

        if isinstance(end_times, (int, float)):
            end_times = np.array([end_times], dtype=float)

        self._model.set_data(events, end_times)

    def incremental_fit(self, events, end_time=None):
        """Incrementally fit model with data by adding one Hawkes realization.

        Parameters
        ----------
        events : `list` of `np.ndarray`
            The events of each component of the realization. Namely
            `events[j]` contains a one-dimensional `np.ndarray` of
            the events' timestamps of component j

        end_time : `float`, default=None
            End time of the realization.
            If None, it will be set to realization's latest time.

        Notes
        -----
        Data is not stored, so this might be useful if the list of all
        realizations does not fit in memory
        """
        if end_time is None:
            end_time = max(map(max, events))

        self._model.incremental_set_data(events, end_time)

        self._set("_fitted", True)
        self._set(N_CALLS_LOSS, 0)
        self._set(PASS_OVER_DATA, 0)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._model.grad(coeffs, out)
        return out

    def hessian(self, x):
        """Return model's hessian

        Parameters
        ----------
        x : `np.ndarray`, shape=(n_coeffs,)
            Value at which the hessian is computed

        Notes
        -----
        For `ModelHawkesExpKernLeastSq` the value of the hessian
        does not depend on the value at which it is computed.
        """
        if not hasattr(self._model, "hessian"):
            raise NotImplementedError('hessian is not implemented yet for '
                                      'this model')

        if not self._fitted:
            raise ValueError("call ``fit`` before using ``hessian``")

        # What kind of integers does scipy use fr sparse indices?
        sparse_dtype = sputils.get_index_dtype()

        n_baselines = self.n_nodes
        # number of alphas per dimension
        if isinstance(
                self._model,
            (ModelHawkesSumExpKernLeastSq, ModelHawkesSumExpKernLogLik)):
            n_alphas_i = self.n_nodes * len(self.decays)
        else:
            n_alphas_i = self.n_nodes

        dim = self.n_nodes
        row_indices_size = n_baselines + dim * n_alphas_i + 1
        data_size = (n_baselines + dim * n_alphas_i) * (1 + n_alphas_i)

        # looks like [0  3  6  9 12 15 18] in dimension 2
        row_indices = np.arange(row_indices_size,
                                dtype=sparse_dtype) * (1 + n_alphas_i)

        # looks like [0 2 3 1 4 5 0 2 3 0 2 3 1 4 5 1 4 5] in dimension 2
        # We first create the recurrent pattern for each dim
        block_dim = {}
        for d in range(dim):
            mu_array = np.array(d)
            alpha_array = n_baselines + d * n_alphas_i + np.arange(n_alphas_i)
            block_dim[d] = np.hstack((mu_array, alpha_array))

        # and then fill the indices array
        indices = np.zeros(data_size, dtype=sparse_dtype)
        for d in range(dim):
            indices[d * (n_alphas_i + 1): (d + 1) * (n_alphas_i + 1)] = \
                block_dim[d]
            alpha_shift = n_baselines * (n_alphas_i + 1)
            alpha_d_start = alpha_shift + d * n_alphas_i * (n_alphas_i + 1)
            alpha_d_end = alpha_shift + (d + 1) * n_alphas_i * (n_alphas_i + 1)
            indices[alpha_d_start: alpha_d_end] = \
                np.tile(block_dim[d], (n_alphas_i,))

        data = np.zeros(data_size, dtype=float)

        # In these two models, hessian does not depend on x
        if isinstance(
                self._model,
            (ModelHawkesSumExpKernLeastSq, ModelHawkesExpKernLeastSq)):
            self._model.hessian(data)
        else:
            self._model.hessian(x, data)

        hessian = csr_matrix((data, indices, row_indices))
        return hessian

    @property
    def n_jumps(self):
        return self._model.get_n_total_jumps()

    @property
    def n_nodes(self):
        return self._model.get_n_nodes()

    @property
    def end_times(self):
        if self._end_times is not None or not self._fitted:
            return self._end_times
        else:
            return self._model.get_end_times()

    @end_times.setter
    def end_times(self, val):
        if self._fitted:
            raise RuntimeError("You cannot set end_times once model has been "
                               "fitted")
        self._set('_end_times', val)
