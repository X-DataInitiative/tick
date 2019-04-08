# License: BSD 3 clause

from abc import abstractmethod
from time import time

from tick.base import Base
from tick.solver.history import History


class Solver(Base):
    """
    The base class for a solver. In only deals with verbosing
    information, creating an History object, etc.

    Parameters
    ----------
    tol : `float`, default=0
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default = 10
        Print history information every time the iteration number is a
        multiple of ``print_every``

    record_every : `int`, default = 1
        Information along iteration is recorded in history each time the
        iteration number of a multiple of ``record_every``

    Attributes
    ----------
    time_start : `str`
        Start date of the call to solve()

    time_elapsed : `float`
        Duration of the call to solve(), in seconds

    time_end : `str`
        End date of the call to solve()

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "history": {
            "writable": False
        },
        "solution": {
            "writable": False
        },
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
        },
        "_record_every": { }
    }

    def __init__(self, tol=0., max_iter=100, verbose=True, print_every=10,
                 record_every=1):
        Base.__init__(self)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.record_every = record_every
        # Create an history object which deals with printing information
        # along the optimization loop, and stores information
        self.history = History()
        self.time_start = None
        self._time_start = None
        self.time_elapsed = None
        self.time_end = None
        self.solution = None

    def _start_solve(self):
        # Reset history
        self.history._clear()
        self._set("time_start", self._get_now())
        self._set("_time_start", time())
        if self.verbose:
            print("Launching the solver " + self.name + "...")

    def _end_solve(self):
        t = time()
        self._set("time_elapsed", t - self._time_start)
        if self.verbose:
            print("Done solving using " + self.name + " in " +
                  str(self.time_elapsed) + " seconds")

    def solve(self, *args, **kwargs):
        self._start_solve()
        self._solve(*args, **kwargs)
        self._end_solve()
        return self.solution

    def _should_record_iter(self, n_iter):
        """Should solver record this iteration or not?
        """
        # If we are never supposed to record
        if self.max_iter < self.print_every and \
                self.max_iter < self.record_every:
            return False
        # Otherwise check that we are either at a specific moment or at the end
        elif n_iter % self.print_every == 0 or n_iter % self.record_every == 0:
            return True
        elif n_iter + 1 == self.max_iter:
            return True
        return False

    def _handle_history(self, n_iter: int, force: bool = False, **kwargs):
        """Handles history for keywords and current iteration

        Parameters
        ----------
        n_iter : `int`
            The current iteration (will determine if we record it or
            not)
        force : `bool`
            If True, we will record no matter the value of ``n_iter``

        **kwargs : `dict`
            key, value pairs of the values to record in the History of
            the solver
        """

        # TODO: this should be protected : _handle_history
        verbose = self.verbose
        print_every = self.print_every
        record_every = self.record_every
        should_print = verbose and (force or n_iter % print_every == 0)
        should_record = force or n_iter % print_every == 0 or \
                        n_iter % record_every == 0
        if should_record:
            iter_time = kwargs.get('iter_time', time() - self._time_start)
            self.history._update(n_iter=n_iter, time=iter_time,
                                 **kwargs)
        if should_print:
            self.history._print_history()

    def print_history(self):
        self.history.print_full_history()

    @abstractmethod
    def _solve(self, *args, **kwargs):
        """Method to be overloaded of the child solver
        """
        pass

    @abstractmethod
    def objective(self, coeffs, loss: float = None):
        """Compute the objective minimized by the solver at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The objective is computed at this point

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """

    def get_history(self, key=None):
        """Returns history of the solver

        Parameters
        ----------
        key : `str`, default=None
            * If `None` all history is returned as a `dict`
            * If `str`, name of the history element to retrieve

        Returns
        -------
        output : `list` or `dict`
            * If ``key`` is None or ``key`` is not in history then
              output is a dict containing history of all keys
            * If ``key`` is the name of an element in the history,
              output is a `list` containing the history of this element
        """
        val = self.history.values.get(key, None)
        if val is None:
            return self.history.values
        else:
            return val

    @property
    def record_every(self):
        return self._record_every

    @record_every.setter
    def record_every(self, val):
        self._record_every = val

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("coeffs", None)
        dd.pop("history", None)
        return dd
