import numpy as np

from tick.optim.prox.base import Prox
from tick.optim.solver.base import SolverFirstOrder
from tick.optim.solver.base.utils import relative_distance


class CompositeProx(Prox):
    """Class of prox that wraps a list of prox

    Parameters
    ----------
    prox_list : List[Prox]
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
        # Range is stored in each Prox individually
        Prox.__init__(self, range=None)
        if len(prox_list) == 0:
            raise ValueError('prox_list must have at least one Prox')
        self.prox_list = prox_list
        self.n_proxs = len(self.prox_list)

    def _call(self, coeffs: np.ndarray, step: object,
              out: np.ndarray):
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


class GFB(SolverFirstOrder):
    """Generalized Forward-Backward algorithm
    
    Minimize the objective

    .. math:: f(x) + sum_i g_i(x)

    using generalized forward-backward. This algorithm assumes that f and all
    g_i are convex, that f has a Lipschitz gradient and that all g_i are
    prox-capable.

    Parameters
    ----------
    step : `float` default=None
        Step-size of the algorithm.

    tol : `float`, default=0.
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=1000
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=1
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    surrelax : `float`, default=1
        Relaxation parameter
    """

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 1000, surrelax=1., verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
        SolverFirstOrder.__init__(self, step=step, tol=tol,
                                  max_iter=max_iter, verbose=verbose,
                                  print_every=print_every,
                                  record_every=record_every)
        self.surrelax = surrelax

    def set_prox(self, prox: list):
        """
        Parameters
        ----------
        prox : list[Prox]
            List of all proximal operators of the model
        """
        prox = CompositeProx(prox)
        SolverFirstOrder.set_prox(self, prox)
        return self

    def initialize_values(self, x0, step):
        step, obj, x, x_old = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=1)
        z_list = [np.zeros_like(x) for _ in range(self.prox.n_proxs)]
        z_old_list = [np.zeros_like(x) for _ in range(self.prox.n_proxs)]
        return x, x_old, z_list, z_old_list, obj, step

    def _solve(self, x0: np.ndarray, step: float):
        x, x_old, z_list, z_old_list, obj, step = \
            self.initialize_values(x0, step)

        n_prox = self.prox.n_proxs
        for n_iter in range(self.max_iter + 1):
            obj_old = obj
            grad_x = self.model.grad(x)
            for i in range(n_prox):
                z = z_list[i]
                z_old = z_old_list[i]
                z[:] = self.prox.call_i(i, 2 * x_old - z_old - step * grad_x,
                                        n_prox * step)
                # Relaxation step
                z[:] = z_old + self.surrelax * (z - x_old)

            x[:] = 1./n_prox * sum(z_list)
            rel_delta = relative_distance(x, x_old)
            obj = self.objective(x)
            rel_obj = abs(obj - obj_old) / abs(obj_old)

            x_old[:] = x
            for i in range(n_prox):
                z_old_list[i][:] = z_list[i]

            converged = rel_obj < self.tol
            # if converged, we stop the loop and record the last step
            # in history
            self._handle_history(n_iter, force=converged, obj=obj,
                                 x=x.copy(), rel_delta=rel_delta,
                                 step=step, rel_obj=rel_obj)
            if converged:
                break

        self._set('solution', x)
