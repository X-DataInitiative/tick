# License: BSD 3 clause

import numpy as np

from tick.prox.base import Prox
from .base import SolverFirstOrder
from .base.utils import relative_distance


class CompositeProx(Prox):
    """Class of prox that wraps a list of prox

    Parameters
    ----------
    prox_list : `list` of `Prox`
        List of prox that are wrapped

    Attributes
    ----------
    n_proxs : int
        Number of wrapped Prox

    Notes
    -----
    You cannot call globally this Prox, you must call all wrapped Prox one
    by one. Otherwise the order of your prox in the ProxList might change the
    result
    """
    _attrinfos = {
        "prox_list": {
            "writable": False
        },
        "n_proxs": {
            "writable": False
        }
    }

    def __init__(self, prox_list: list):
        if isinstance(prox_list, Prox):
            prox_list = [prox_list]

        # Range is stored in each Prox individually
        Prox.__init__(self, range=None)
        if len(prox_list) == 0:
            raise ValueError('prox_list must have at least one Prox')
        self.prox_list = prox_list
        self.n_proxs = len(self.prox_list)

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        raise ValueError("You cannot call globally a CompositeProx")

    def call_i(self, i: int, coeffs: np.ndarray, step: object):
        """Calls ith prox
        """
        return self.prox_list[i].call(coeffs, step)

    def value(self, coeffs: np.ndarray):
        prox_value = 0
        for prox in self.prox_list:
            prox_value += prox.value(coeffs)
        return prox_value

    def astype(self, dtype_or_object_with_dtype):
        def cast_prox(prox):
            return prox.astype(dtype_or_object_with_dtype)

        return CompositeProx(list(map(cast_prox, self.prox_list)))


class GFB(SolverFirstOrder):
    """Generalized Forward-Backward algorithm

    For the minimization of objectives of the form

    .. math::
        f(x) + \\sum_{p=1}^P g_p(x)

    where :math:`f` has a smooth gradient and :math:`g_1, \ldots, g_P` are
    prox-capable. Function :math:`f` corresponds to the ``model.loss`` method
    of the model (passed with ``set_model`` to the solver) and
    :math:`g_1, \ldots, g_P` correspond to the list of prox passed with the
    ``set_prox`` method.
    One iteration of :class:`GFB <tick.solver.GFB>` is as follows:

    .. math::
        \\begin{align*}
        &\\text{for } p=1, \\ldots, P \\; \\text{ do the following:} \\\\
        & \\quad z_p \\gets \\mathrm{prox}_{P \\eta g_p} \\Big(2 w - z_p^{\\text{old}}
        - \\eta \\nabla f(w) \\Big) \\\\
        & \\quad z_p \\gets z_p^{\\text{old}} + \\beta (z_p - w) \\\\
        &w \\gets \\frac 1P \\sum_{p=1}^P z_p
        \\end{align*}

    where :math:`\\nabla f(w)` is the gradient of :math:`f` given by the
    ``model.grad`` method and :math:`\\mathrm{prox}_{\\eta g_p}` is given by the
    ``prox[p].call`` method. The step-size :math:`\\eta` can be tuned with
    ``step``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver. The level of sur-relaxation :math:`\\beta` can be tuned
    using the ``surrelax`` attribute.

    Parameters
    ----------
    step : `float`, default=None
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be tuned as
        ``step = 1 / model.get_lip_best()``

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=500
        Maximum number of iterations of the solver.

    surrelax : `float`, default=1
        Level of sur-relaxation to use in the algorithm.

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `list` of `Prox`
        List of proximal operators used by the solver, passed with the
        ``set_prox`` method

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
    * H. Raguet, J. Fadili, G. Peyré, A generalized forward-backward splitting,
      *SIAM Journal on Imaging Sciences* (2013)
    """

    def __init__(self, step: float = None, tol: float = 1e-10,
                 max_iter: int = 500, surrelax=1., verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self.surrelax = surrelax

    def set_prox(self, prox_list: list):
        """
        Parameters
        ----------
        prox_list : `list` of `Prox`
            List of all proximal operators of the model
        """
        if not isinstance(prox_list, CompositeProx):
            prox_list = CompositeProx(prox_list)

        if self.dtype is not None:
            prox_list = prox_list.astype(self.dtype)
        SolverFirstOrder.set_prox(self, prox_list)
        return self

    def initialize_values(self, x0, step):
        step, obj, x, x_old = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=1)
        z_list = [np.zeros_like(x) for _ in range(self.prox.n_proxs)]
        z_old_list = [np.zeros_like(x) for _ in range(self.prox.n_proxs)]
        return x, x_old, z_list, z_old_list, obj, step

    def _solve(self, x0: np.ndarray, step: float):
        minimizer, prev_minimizer, z_list, z_old_list, obj, step = \
            self.initialize_values(x0, step)

        n_prox = self.prox.n_proxs
        for n_iter in range(self.max_iter):
            # We will record on this iteration and we must be ready
            if self._should_record_iter(n_iter):
                prev_minimizer[:] = minimizer
                prev_obj = self.objective(prev_minimizer)

            grad_x = self.model.grad(minimizer)
            for i in range(n_prox):
                z = z_list[i]
                z_old = z_old_list[i]
                z[:] = self.prox.call_i(
                    i, 2 * prev_minimizer - z_old - step * grad_x,
                    n_prox * step)
                # Relaxation step
                z[:] = z_old + self.surrelax * (z - prev_minimizer)

            minimizer[:] = 1. / n_prox * sum(z_list)

            for i in range(n_prox):
                z_old_list[i][:] = z_list[i]

            # Let's record metrics
            if self._should_record_iter(n_iter):
                rel_delta = relative_distance(minimizer, prev_minimizer)
                obj = self.objective(minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)

                converged = rel_obj < self.tol
                # if converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     step=step, rel_obj=rel_obj)
                if converged:
                    break

        self._set('solution', minimizer)
