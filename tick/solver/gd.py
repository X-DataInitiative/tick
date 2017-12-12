# License: BSD 3 clause

import numpy as np
from numpy.linalg import norm

from .base import SolverFirstOrder
from .base.utils import relative_distance


class GD(SolverFirstOrder):
    """Proximal gradient descent

    For the minimization of objectives of the form

    .. math::
        f(w) + g(w),

    where :math:`f` has a smooth gradient and :math:`g` is prox-capable.
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    One iteration of :class:`GD <tick.solver.GD>` is as follows:

    .. math::
        w \\gets \\mathrm{prox}_{\\eta g} \\big(w - \\eta \\nabla f(w) \\big),

    where :math:`\\nabla f(w)` is the gradient of :math:`f` given by the
    ``model.grad`` method and :math:`\\mathrm{prox}_{\\eta g}` is given by the
    ``prox.call`` method. The step-size :math:`\\eta` can be tuned with
    ``step``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver.

    Parameters
    ----------
    step : `float`, default=None
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_best()``. If ``linesearch=True``, this is
        the first step-size to be used in the linesearch (that should be taken
        as too large).

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver.

    linesearch : `bool`, default=True
        If `True`, use backtracking linesearch to tune the step automatically.

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    linesearch_step_increase : `float`, default=2.
        Factor of step increase when using linesearch

    linesearch_step_decrease : `float`, default=0.5
        Factor of step decrease when using linesearch

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    References
    ----------
    * A. Beck and M. Teboulle, A fast iterative shrinkage-thresholding
      algorithm for linear inverse problems,
      *SIAM journal on imaging sciences*, 2009
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
