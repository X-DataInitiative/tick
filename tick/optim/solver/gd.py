import numpy as np
from numpy.linalg import norm

from tick.optim.solver.base import SolverFirstOrder
from tick.optim.solver.base.utils import relative_distance


class GD(SolverFirstOrder):
    """
    GD (proximal gradient descent) algorithm.

    Parameters
    ----------
    step : `float` default=None
        Step-size of the algorithm. If ``linesearch=True``, this is the
        first step-size to be used in the linesearch
        (typically taken too large). Otherwise, it's the constant step
        to be used along iterations.

    tol : `float`, default=0.
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    linesearch : `bool`, default=True
        Use backtracking linesearch

    linesearch_step_increase : `float`, default=2.
        Factor of step increase when using linesearch

    linesearch_step_decrease : `float`, default=0.5
        Factor of step decrease when using linesearch

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=1
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``
    """

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 100, linesearch: bool = True,
                 linesearch_step_increase: float = 2.,
                 linesearch_step_decrease: float = 0.5,
                 verbose: bool = True, print_every: int = 10,
                 record_every: int = 1):
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self.linesearch = linesearch
        self.linesearch_step_increase = linesearch_step_increase
        self.linesearch_step_decrease = linesearch_step_decrease

    def _initialize_values(self, x0=None, step=None):
        if step is None:
            if self.step is None:
                if self.linesearch:
                    # If we use linesearch, then we can choose a large
                    # initial step
                    step = 1e9
                else:
                    raise ValueError("No step specified.")
        step, obj, x, prev_x, x_new = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=2)
        return x, prev_x, x_new, step, obj

    def _gradient_step(self, x, x_new, step):
        if self.linesearch:
            step *= self.linesearch_step_increase
            loss_x, grad_x = self.model.loss_and_grad(x)
            obj_x = self.objective(x, loss=loss_x)
            while True:
                x_new[:] = self.prox.call(x - step * grad_x, step)
                obj_x_new = self.objective(x_new)
                envelope = obj_x + np.sum(grad_x * (x_new - x),
                                          axis=None) \
                           + 1. / (2 * step) * norm(x_new - x) ** 2
                test = (obj_x_new <= envelope)
                if test:
                    break
                step *= self.linesearch_step_decrease
                if step == 0:
                    break
        else:
            grad_x = self.model.grad(x)
            x_new[:] = self.prox.call(x - step * grad_x, step)
            obj_x_new = self.objective(x_new)
        x[:] = x_new
        return x, step, obj_x_new

    def _solve(self, x0: np.ndarray = None, step: float = None):
        x, prev_x, x_new, step, obj = self._initialize_values(x0, step)
        for n_iter in range(self.max_iter + 1):
            prev_x[:] = x
            prev_obj = obj
            x, step, obj = self._gradient_step(x, x_new, step)
            # x, y, t, step = self._gradient_step(x, prev_x, y, grad_y, t,
            #                                     prev_t, step)
            if step == 0:
                print('Step equals 0... at %i' % n_iter)
                break
            rel_delta = relative_distance(x, prev_x)
            obj = self.objective(x)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            converged = rel_obj < self.tol
            # If converged, we stop the loop and record the last step
            # in history
            self._handle_history(n_iter, force=converged, obj=obj,
                                 x=x.copy(), rel_delta=rel_delta,
                                 step=step, rel_obj=rel_obj)
            if converged:
                break
        self._set("solution", x)
        return x
