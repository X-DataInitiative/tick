# License: BSD 3 clause

import numpy as np

from tick.base_model import Model
from tick.prox.base import Prox
from . import SolverFirstOrder, SolverSto
from .utils import relative_distance

__author__ = 'stephanegaiffas'


# TODO: property for step that sets it in the C++


class SolverFirstOrderSto(SolverFirstOrder, SolverSto):
    """The base class for a first order stochastic solver.
    It only deals with verbosing information, and setting parameters.

    Parameters
    ----------
    epoch_size : `int`
        Epoch size

    rand_type : `str`
        Type of random sampling

        * if ``"unif"`` samples are uniformly drawn among all possibilities
        * if ``"perm"`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

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

    seed : `int`
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    Attributes
    ----------
    model : `Solver`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

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
        "_step": {
            "writable": False
        }
    }

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type="unif", tol=0., max_iter=100, verbose=True,
                 print_every=10, record_every=1, seed=-1):

        self._step = None

        # We must first construct SolverSto (otherwise self.step won't
        # work in SolverFirstOrder)
        SolverSto.__init__(self, epoch_size=epoch_size, rand_type=rand_type,
                           seed=seed)
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)

    def set_model(self, model: Model):
        """Set model in the solver

        Parameters
        ----------
        model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The `Solver` with given model
        """
        SolverFirstOrder.set_model(self, model)
        SolverSto.set_model(self, model)
        return self

    def set_prox(self, prox: Prox):
        """Set proximal operator in the solver

        Parameters
        ----------
        prox : `Prox`
            The proximal operator of the penalization function

        Returns
        -------
        output : `Solver`
            The solver with given prox

        Notes
        -----
        In some solvers, ``set_model`` must be called before
        ``set_prox``, otherwise and error might be raised
        """
        SolverFirstOrder.set_prox(self, prox)
        SolverSto.set_prox(self, prox)
        return self

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        self._set("_step", val)
        if val is None:
            val = 0.
        if self._solver is not None:
            self._solver.set_step(val)

    def _solve(self, x0: np.array = None, step: float = None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : np.array, shape=(n_coeffs,)
            Starting iterate for the solver

        step : float
            Step-size or learning rate for the solver

        Returns
        -------
        output : np.array, shape=(n_coeffs,)
            Obtained minimizer
        """
        from tick.solver import SDCA
        if not isinstance(self, SDCA):
            if step is not None:
                self.step = step

            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(x0, step, n_empty_vectors=1)
            self._solver.set_starting_iterate(minimizer)

        else:
            # In sdca case x0 is a dual vector
            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(None, step, n_empty_vectors=1)
            if x0 is not None:
                self._solver.set_starting_iterate(x0)

        # At each iteration we call self._solver.solve that does a full
        # epoch
        for n_iter in range(self.max_iter + 1):
            prev_minimizer[:] = minimizer
            prev_obj = obj
            # Launch one epoch using the wrapped C++ solver
            self._solver.solve()
            self._solver.get_minimizer(minimizer)
            # The step might be modified by the C++ solver
            # step = self._solver.get_step()
            obj = self.objective(minimizer)
            rel_delta = relative_distance(minimizer, prev_minimizer)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            converged = rel_obj < self.tol
            # If converged, we stop the loop and record the last step
            # in history
            self._handle_history(n_iter, force=converged, obj=obj,
                                 x=minimizer.copy(), rel_delta=rel_delta,
                                 rel_obj=rel_obj)
            if converged:
                break
        self._set("solution", minimizer)
        return minimizer
