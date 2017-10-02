import numpy as np

from tick.optim.proj import ProjHalfSpace
from tick.optim.prox.base import Prox
from tick.optim.prox import ProxZero, ProxL2Sq
from tick.optim.solver.base import SolverFirstOrder
from tick.optim.solver.base.utils import relative_distance


class Newton(SolverFirstOrder):
    """Newton descent

    Parameters
    ----------
    tol : `float`, default=0.
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
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

    References
    ----------
    Quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS), see
    Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.
    """

    _attrinfos = {
        "_prox_grad": {
            "writable": False
        },
        "_prox_hess": {
            "writable": False
        },
        '_proj': {}
    }

    def __init__(self, tol: float = 1e-10,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
        SolverFirstOrder.__init__(self, step=None, tol=tol,
                                  max_iter=max_iter, verbose=verbose,
                                  print_every=print_every,
                                  record_every=record_every)
        self._prox_grad = None

    def set_model(self, model):
        A = model.features
        # mask = model.labels > 0
        A = A[model.labels > 0, :]
        b = 1e-8 + np.zeros(A.shape[0])
        self._set('_proj', ProjHalfSpace(max_iter=1000).fit(A, b))
        return SolverFirstOrder.set_model(self, model)

    def set_prox(self, prox: Prox):
        """Set proximal operator in the solver.

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
        ``set_prox``, otherwise and error might be raised.
        """
        if type(prox) is ProxZero:
            SolverFirstOrder.set_prox(self, prox)
            self._set("_prox_grad", lambda x: np.zeros_like(x))
            self._set("_prox_hess", lambda x: np.zeros(len(x), len(x)))
        elif type(prox) is ProxL2Sq:
            SolverFirstOrder.set_prox(self, prox)
            self._set("_prox_grad", lambda x: prox.strength * x)
            self._set("_prox_hess", lambda x: prox.strength *
                                              np.identity(len(x)))
        else:
            raise ValueError("Newton only accepts ProxZero and ProxL2sq "
                             "for now")

        return self

    def solve(self, x0=None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : `np.array`, shape=(n_coeffs,), default=`None`
            Starting point of the solver

        Returns
        -------
        output : `np.array`, shape=(n_coeffs,)
            Obtained minimizer for the problem
        """
        self._start_solve()
        coeffs = self._solve(x0)
        self._set("solution", coeffs)
        self._end_solve()
        return self.solution

    def _solve(self, x0: np.ndarray = None):

        if x0 is not None:
            x = x0.copy()
        else:
            x = np.zeros(self.model.n_coeffs)

        x = self._proj.call(x)
        obj = self.objective(x)


        assert self.model.features[self.model.labels > 0, :].dot(x).min() > 0

        prev_x = np.empty_like(x)
        for n_iter in range(self.max_iter + 1):
            prev_x[:] = x
            prev_obj = obj

            hessian = self.model.hessian(x) + self._prox_hess(x)
            grad = self.model.grad(x) + self._prox_grad(x)

            step = 1
            beta = 0.7
            direction = np.linalg.inv(hessian).dot(grad)
            while True:
                next_x = x - step * direction
                next_objective = self.objective(next_x)
                if np.isnan(next_objective) or (next_objective > prev_obj):
                    step *= beta
                else:
                    x = next_x
                    break


            rel_delta = relative_distance(x, prev_x)

            obj = self.objective(x)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            converged = rel_obj < self.tol

            # If converged, we stop the loop and record the last step
            # in history
            self._handle_history(n_iter, force=converged, obj=obj,
                                 x=x.copy(), rel_delta=rel_delta,
                                 rel_obj=rel_obj)
            if converged:
                break
        self._set("solution", x)
        return x
