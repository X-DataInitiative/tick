# License: BSD 3 clause

import numpy as np
from numpy.linalg import norm

from .base import SolverFirstOrder
from .base.utils import relative_distance


class AGD(SolverFirstOrder):
    """Accelerated proximal gradient descent

    For the minimization of objectives of the form

    .. math::
        f(w) + g(w),

    where :math:`f` has a smooth gradient and :math:`g` is prox-capable.
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    One iteration of :class:`AGD <tick.solver.AGD>` is as follows:

    .. math::
        w^{k} &\\gets \\mathrm{prox}_{\\eta g} \\big(z^k - \\eta \\nabla f(z^k)
        \\big) \\\\
        t_{k+1} &\\gets \\frac{1 + \sqrt{1 + 4 t_k^2}}{2} \\\\
        z^{k+1} &\\gets w^k + \\frac{t_k - 1}{t_{k+1}} (w^k - w^{k-1})

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

    def __init__(self, step: float = None, tol: float = 1e-10,
                 max_iter: int = 100, linesearch: bool = True,
                 linesearch_step_increase: float = 2.,
                 linesearch_step_decrease: float = 0.5, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
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
        step, obj, x, prev_x, grad_y = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=2)
        y = x.copy()
        t = 1.
        return x, prev_x, y, grad_y, t, step, obj

    def _gradient_step(self, x, prev_x, y, grad_y, t, prev_t, step):
        if self.linesearch:
            step *= self.linesearch_step_increase
            loss_y, _ = self.model.loss_and_grad(y, out=grad_y)
            obj_y = self.objective(y, loss=loss_y)
            while True:
                x[:] = self.prox.call(y - step * grad_y, step)
                obj_x = self.objective(x)
                envelope = obj_y + np.sum(grad_y * (x - y), axis=None) \
                           + 1. / (2 * step) * norm(x - y) ** 2
                test = (obj_x <= envelope)
                if test:
                    break
                step *= self.linesearch_step_decrease
                if step == 0:
                    break
        else:
            grad_y = self.model.grad(y)
            x[:] = self.prox.call(y - step * grad_y, step)
        t = np.sqrt((1. + (1. + 4. * t * t))) / 2.
        y[:] = x + (prev_t - 1) / t * (x - prev_x)
        return x, y, t, step

    def _solve(self, x0: np.ndarray = None, step: float = None):
        minimizer, prev_minimizer, y, grad_y, t, step, obj = \
            self._initialize_values(x0, step)
        for n_iter in range(self.max_iter):
            prev_t = t
            prev_minimizer[:] = minimizer

            # We will record on this iteration and we must be ready
            if self._should_record_iter(n_iter):
                prev_obj = self.objective(prev_minimizer)

            minimizer, y, t, step = self._gradient_step(
                minimizer, prev_minimizer, y, grad_y, t, prev_t, step)
            if step == 0:
                print('Step equals 0... at %i' % n_iter)
                break

            # Let's record metrics
            if self._should_record_iter(n_iter):
                rel_delta = relative_distance(minimizer, prev_minimizer)
                obj = self.objective(minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                converged = rel_obj < self.tol
                # If converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     step=step, rel_obj=rel_obj)
                if converged:
                    break

        self._set("solution", minimizer)
        return minimizer
