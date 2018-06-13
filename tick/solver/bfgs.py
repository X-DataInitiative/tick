# License: BSD 3 clause

import numpy as np
from scipy.optimize import fmin_bfgs

from tick.base_model import Model
from tick.prox import ProxZero, ProxL2Sq
from tick.prox.base import Prox
from .base import SolverFirstOrder
from .base.utils import relative_distance


class BFGS(SolverFirstOrder):
    """Broyden, Fletcher, Goldfarb, and Shanno algorithm

    This solver is actually a simple wrapping of `scipy.optimize.fmin_bfgs`
    BFGS (Broyden, Fletcher, Goldfarb, and Shanno) algorithm. This is a
    quasi-newton algotithm that builds iteratively approximations of the inverse
    Hessian. This solver can be used to minimize objectives of the form

    .. math::
        f(w) + g(w),

    for :math:`f` with a smooth gradient and only :math:`g` corresponding to
    the zero penalization (namely :class:`ProxZero <tick.prox.ProxZero>`)
    or ridge penalization (namely :class:`ProxL2sq <tick.prox.ProxL2sq>`).
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver.

    Parameters
    ----------
    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=10
        Maximum number of iterations of the solver

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

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    References
    ----------
    * Quasi-Newton method of Broyden, Fletcher, Goldfarb and Shanno (BFGS),
      see Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.
    """

    _attrinfos = {"_prox_grad": {"writable": False}}

    def __init__(self, tol: float = 1e-10, max_iter: int = 10,
                 verbose: bool = True, print_every: int = 1,
                 record_every: int = 1):
        SolverFirstOrder.__init__(self, step=None, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self._prox_grad = None

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
            self._set("_prox_grad", lambda x: x)
        elif type(prox) is ProxL2Sq:
            SolverFirstOrder.set_prox(self, prox)
            self._set("_prox_grad", lambda x: prox.strength * x)
        else:
            raise ValueError("BFGS only accepts ProxZero and ProxL2sq "
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
        if x0 is None:
            x0 = np.zeros(self.model.n_coeffs, dtype=self.dtype)
        obj = self.objective(x0)

        # A closure to maintain history along internal BFGS's iterations
        n_iter = [0]
        prev_x = x0.copy()

        def insp(xk):
            if self._should_record_iter(n_iter[0]):
                prev_obj = self.objective(prev_x)
                x = xk
                rel_delta = relative_distance(x, prev_x)

                obj = self.objective(x)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                self._handle_history(n_iter[0], force=False, obj=obj,
                                     x=xk.copy(), rel_delta=rel_delta,
                                     rel_obj=rel_obj)
            prev_x[:] = xk
            n_iter[0] += 1

        insp.n_iter = n_iter
        insp.self = self
        insp.prev_x = prev_x

        # We simply call the scipy.optimize.fmin_bfgs routine
        x_min, f_min, _, _, _, _, _ = \
            fmin_bfgs(lambda x: self.model.loss(x) + self.prox.value(x),
                      x0,
                      lambda x: self.model.grad(x) + self._prox_grad(x),
                      maxiter=self.max_iter, gtol=self.tol,
                      callback=insp, full_output=True,
                      disp=False, retall=False)

        return x_min

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
        self.dtype = model.dtype
        if np.dtype(self.dtype) != np.dtype("float64"):
            raise ValueError(
                "Solver BFGS currenty only accepts float64 array types")
        return SolverFirstOrder.set_model(self, model)
