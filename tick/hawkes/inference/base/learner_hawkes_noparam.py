# License: BSD 3 clause

import numpy as np

from tick.solver.base import Solver


class LearnerHawkesNoParam(Solver):
    """Base class of Hawkes learners

    Parameters
    ----------
    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    _attrinfos = {
        "_fitted": {
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
        "_learner": {
            "writable": False
        }
    }

    _cpp_obj_name = '_learner'

    def __init__(self, tol=1e-5, verbose=False, approx=0, n_threads=1,
                 max_iter=100, print_every=10, record_every=10):
        Solver.__init__(self, tol=tol, verbose=verbose, max_iter=max_iter,
                        print_every=print_every, record_every=record_every)

        self.approx = approx
        self.n_threads = n_threads
        self.verbose = verbose
        self.data = None
        self._end_times = None
        self._fitted = False
        self._learner = None

    def _get_n_coeffs(self):
        return self._learner.get_n_coeffs()

    def fit(self, events, end_times=None):
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
        self._set('_fitted', True)
        self._set('_end_times', end_times)
        self._set_data(events)
        return self

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

        events, end_times = self._clean_events_and_endtimes(events)

        try:
            self._learner.set_data(events, end_times)
        except TypeError:
            self._learner.set_data(events)

    def _clean_events_and_endtimes(self, events):
        if len(events[0]) == 0 or not isinstance(events[0][0], np.ndarray):
            events = [events]

        end_times = self._end_times
        if end_times is None:
            non_empty_events = [[r for r in e if len(r) > 0] for e in events]
            end_times = np.array([max(map(max, e)) for e in non_empty_events])

        if isinstance(end_times, (int, float)):
            end_times = np.array([end_times], dtype=float)

        return events, end_times

    @property
    def n_jumps(self):
        return self._learner.get_n_total_jumps()

    @property
    def n_nodes(self):
        return self._learner.get_n_nodes()

    @property
    def n_realizations(self):
        return len(self._learner.get_n_jumps_per_realization())

    @property
    def end_times(self):
        if self._end_times is not None or not self._fitted:
            return self._end_times
        else:
            return self._learner.get_end_times()

    @end_times.setter
    def end_times(self, val):
        if self._fitted:
            raise RuntimeError("You cannot set end_times once model has been "
                               "fitted")
        self._set('_end_times', val)
