# License: BSD 3 clause

import numpy as np
from abc import ABC, abstractmethod

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

    _attrinfos = {"_step": {"writable": False}}

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type="unif", tol: float = 0., max_iter=100, verbose=True,
                 print_every=10, record_every=1, seed=-1):

        self._step = None

        # We must first construct SolverSto (otherwise self.step won't
        # work in SolverFirstOrder)
        SolverSto.__init__(self, epoch_size=epoch_size, rand_type=rand_type,
                           seed=seed)
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)

        self._set_cpp_solver('float64')

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
        self.validate_model(model)
        if self.dtype != model.dtype or self._solver is None:
            self._set_cpp_solver(model.dtype)

        self.dtype = model.dtype
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

    @property
    def record_every(self):
        if hasattr(self, '_solver') and self._solver is not None:
            return self._solver.get_record_every()
        else:
            return self._record_every

    @record_every.setter
    def record_every(self, val):
        self._record_every = val
        if hasattr(self, '_solver') and self._solver is not None:
            self._solver.set_record_every(val)

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

        if self.verbose or self.tol != 0:
            self._solve_with_printing(prev_minimizer, minimizer)
        else:
            self._solve_and_record_in_cpp(minimizer)

        self._solver.get_minimizer(minimizer)
        self._set("solution", minimizer)
        return minimizer

    def _solve_with_printing(self, prev_minimizer, minimizer):
        # At each iteration we call self._solver.solve that does a full
        # epoch
        prev_minimizer[:] = minimizer
        prev_obj = self.objective(prev_minimizer)

        for n_iter in range(self.max_iter):

            # Launch one epoch using the wrapped C++ solver
            self._solver.solve()

            # Let's record metrics
            if self._should_record_iter(n_iter):
                self._solver.get_minimizer(minimizer)
                # The step might be modified by the C++ solver
                # step = self._solver.get_step()
                obj = self.objective(minimizer)
                rel_delta = relative_distance(minimizer, prev_minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj) \
                    if prev_obj != 0 else abs(obj)
                converged = rel_obj < self.tol
                # If converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     rel_obj=rel_obj)
                prev_minimizer[:] = minimizer
                prev_obj = self.objective(prev_minimizer)
                if converged:
                    break

    def _solve_and_record_in_cpp(self, minimizer):
        prev_obj = self.objective(minimizer)
        self._solver.set_prev_obj(prev_obj)
        self._solver.solve(self.max_iter)
        self._post_solve_and_record_in_cpp(minimizer, prev_obj)

    def _post_solve_and_record_in_cpp(self, minimizer, prev_obj):
        prev_iterate = minimizer
        for epoch, iter_time, iterate, obj in zip(
                self._solver.get_epoch_history(),
                self._solver.get_time_history(),
                self._solver.get_iterate_history(),
                self._solver.get_objectives()):
            if epoch is self._solver.get_epoch_history()[-1]:
                # This rel_obj is not exactly the same one as prev_obj is not the
                # objective of the previous epoch but of the previouly recorded
                # epoch
                self._handle_history(
                    epoch, force=True,
                    obj=obj, iter_time=iter_time, x=iterate,
                    rel_delta=relative_distance(iterate, prev_iterate),
                    rel_obj=abs(obj - prev_obj) / abs(prev_obj) \
                        if prev_obj != 0 else abs(obj))
            prev_obj = obj
            prev_iterate[:] = iterate
        minimizer = prev_iterate

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)

    def _extract_dtype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.extract_dtype(
            dtype_or_object_with_dtype)

    @abstractmethod
    def _set_cpp_solver(self, dtype):
        pass

    def astype(self, dtype_or_object_with_dtype):
        if self.model is None:
            raise ValueError("Cannot reassign solver without a model")

        import tick.base.dtype_to_cpp_type
        new_solver = tick.base.dtype_to_cpp_type.copy_with(
            self,
            ["prox", "model", "_solver"]  # ignore on deepcopy
        )
        new_solver._set_cpp_solver(dtype_or_object_with_dtype)
        new_solver.set_model(self.model.astype(new_solver.dtype))
        if self.prox is not None:
            new_solver.set_prox(self.prox.astype(new_solver.dtype))
        return new_solver
