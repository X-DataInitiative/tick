# -*- coding: utf8 -*-

import numpy as np
from numpy.linalg import norm

from tick.optim.model.base import ModelSecondOrder
from tick.optim.prox.base import Prox
from tick.optim.solver.base import SolverFirstOrder
from tick.optim.solver.base.utils import relative_distance

__author__ = 'MartinBompaire'


class SCPG(SolverFirstOrder):
    """
    Minimize the objective

    .. math:: f(x) + g(x)

    using the Proximal Gradient algorithm
    This algorithm assumes that f and g are both convex, that
    f is standard self-concordant and that g is prox-capable.

    Parameters
    ----------

    modified : `bool` (default False)
        Weather or not using the modified version fo the algorithm.

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

    _attrinfos = {
        'model_ssc': {'writable': False},
        'prox_ssc': {'writable': False},
        '_th_gain': {},
        "_initial_n_hessiannorm_calls": {
            "writable": False
        },
    }

    class _ModelStandardSC:
        """The standard self concordant version of the model
        """

        def __init__(self, model: ModelSecondOrder):
            self.original_model = model
            self.sc_constant = model._sc_constant
            self.sc_corr = self.sc_constant ** 2 / 4
            self.sc_corr_sqrt = self.sc_constant / 2
            self._initial_n_hessiannorm_calls = 0

        def loss(self, coeffs: np.ndarray):
            return self.original_model.loss(coeffs) * self.sc_corr

        def grad(self, coeffs: np.ndarray, out: np.ndarray = None):
            out = self.original_model.grad(coeffs, out=out)
            out[:] = out * self.sc_corr
            return out

        def loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray = None):
            loss, out = self.original_model.loss_and_grad(coeffs, out=out)
            out[:] = out * self.sc_corr
            return loss * self.sc_corr, out

        def hessian_norm(self, coeffs: np.ndarray, point: np.ndarray):
            return self.original_model.hessian_norm(coeffs, point) * \
                   self.sc_corr_sqrt

        def original_loss(self, loss_ssc):
            """Returns the loss of the original model form the loss of the
            standard self-concordant model
            """
            return loss_ssc / self.sc_corr

        def original_objective(self, obj_ssc):
            """Returns the loss of the original model form the loss of the
            standard self-concordant model
            """
            return obj_ssc / self.sc_corr

    class _ProxStandardSC:
        """The standard self concordant version of the prox according to the
        model
        """

        def __init__(self):
            self.original_prox = None
            self.sc_constant = None
            self.sc_corr = None

        def set_original_prox(self, prox: Prox):
            self.original_prox = prox

        def set_self_conc_constant(self, self_conc_constant: float):
            self.sc_constant = self_conc_constant
            self.sc_corr = self.sc_constant ** 2 / 4

        def call(self, coeffs: np.ndarray, t: float = 1.,
                 out: np.ndarray = None):
            out = self.original_prox.call(coeffs, step=t * self.sc_corr,
                                          out=out)
            return out

        def value(self, coeffs: np.ndarray):
            return self.original_prox.value(coeffs) * self.sc_corr

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 100, linesearch_step_increase: float = 2.,
                 linesearch_step_decrease: float = 0.5, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1,
                 modified=False):
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)

        self.linesearch_step_increase = linesearch_step_increase
        self.linesearch_step_decrease = linesearch_step_decrease

        self.modified = modified
        self.model_ssc = None
        self.prox_ssc = self._ProxStandardSC()
        self._th_gain = 0

    def set_model(self, model: ModelSecondOrder):
        """Set model in the solver

        Parameters
        ----------
        model : `ModelSecondOrder` and `ModelSelfConcordant`
            Sets the model in the solver. The model gives
            information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The same instance with given model
        """
        self._set('model', model)
        self._set('model_ssc', self._ModelStandardSC(model=model))
        self.prox_ssc.set_self_conc_constant(model._sc_constant)
        return self

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
        """
        self._set('prox', prox)
        self.prox_ssc.set_original_prox(prox)
        return self

    def _objective_ssc(self, x, loss_ssc: float = None):
        """Compute objective at x for the standard self concordant model

        .. math::`\frac{M_f^2}{4} * (f(x) + g(x))`

        Parameters
        ----------
        x : `float`
            The value at which we compute the objective

        loss_ssc: `float`
            The value of the f(x) if it is already known
        """
        if loss_ssc is None:
            return self.model_ssc.loss(x) + self.prox_ssc.value(x)
        else:
            return loss_ssc + self.prox_ssc.value(x)

    def _initialize_values(self, x0, step):
        step, obj, x, prev_x, prev_grad_x_ssc, grad_x_ssc = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=3)

        grad_x_ssc = self.model_ssc.grad(x, out=grad_x_ssc)
        return step, obj, x, prev_x, prev_grad_x_ssc, grad_x_ssc

    def _perform_line_search(self, x, y, step):
        step *= self.linesearch_step_increase
        llh_y_ssc, grad_y_ssc = self.model_ssc.loss_and_grad(y)
        obj_y_ssc = self._objective_ssc(y, loss_ssc=llh_y_ssc)
        while True:

            x[:] = self.prox_ssc.call(y - step * grad_y_ssc, step)
            if norm(x) == 0:
                test = False
            else:
                obj_x_ssc = self._objective_ssc(x)
                envelope = obj_y_ssc + \
                           np.sum(grad_y_ssc * (x - y), axis=None) + \
                           1. / (2 * step) * norm(x - y) ** 2
                test = (obj_x_ssc <= envelope)
            if test:
                break
            step *= self.linesearch_step_decrease
        return step

    def _gradient_step(self, x, prev_x, grad_x_ssc, prev_grad_x_ssc,
                       n_iter, l_k):
        # Testing if our value of l_k fits the condition for the stepsize
        # alpha_k

        # Barzilai-Borwein step
        if n_iter % 10 == 1:
            tmp = x - prev_x
            if np.linalg.norm(tmp, 2) ** 2 > 0:
                l_k = (grad_x_ssc - prev_grad_x_ssc).dot(tmp) / \
                      (np.linalg.norm(tmp, 2) ** 2)
        l_k *= 2

        prev_grad_x_ssc[:] = grad_x_ssc
        prev_x[:] = x
        # Compute x_new, next step vector
        condition = False
        while not condition:
            y_k = self.prox_ssc.call(x - grad_x_ssc / l_k, 1. / l_k)
            d_k = y_k - x
            beta_k = np.sqrt(l_k) * np.linalg.norm(d_k, 2)
            lambda_k = np.sqrt(self.model_ssc.hessian_norm(x, d_k))

            alpha_k = beta_k * beta_k / \
                      (lambda_k * (lambda_k + beta_k * beta_k))
            # condition = lambda_k >= 1 or lambda_k >= beta_k
            condition = 0 <= alpha_k < 1

            if not condition:
                l_k /= 2

            if np.isnan(l_k):
                raise ValueError('l_k is nan')

        self._th_gain = beta_k * beta_k / lambda_k - np.log(1 + beta_k * beta_k
                                                            / lambda_k)
        x_new = x + alpha_k * d_k

        # we also "return" grad_x_ssc and prev_grad_x_ssc which are filled
        # during function's run
        return x_new, y_k, alpha_k, beta_k, lambda_k, l_k

    def _solve(self, x0: np.ndarray = None, step: float = 1e5):

        step, obj, x, prev_x, prev_grad_x_ssc, grad_x_ssc = \
            self._initialize_values(x0, step=step)

        if self.modified:
            grad_y_ssc = np.empty_like(x)

        if self.step is None:
            self.step = 1e5

        step = self._perform_line_search(x, x.copy(), self.step)
        l_k = 1. / step

        for n_iter in range(self.max_iter + 1):

            prev_obj = obj

            x, y, alpha_k, beta_k, lambda_k, l_k = \
                self._gradient_step(x, prev_x, grad_x_ssc,
                                    prev_grad_x_ssc, n_iter, l_k)

            if self.modified:
                llh_y_ssc, _ = self.model_ssc.loss_and_grad(y, out=grad_y_ssc)
                llh_x_ssc, _ = self.model_ssc.loss_and_grad(x, out=grad_x_ssc)

                if self._objective_ssc(y, loss_ssc=llh_y_ssc) < \
                        self._objective_ssc(x, loss_ssc=llh_x_ssc):
                    x[:] = y
                    grad_x_ssc[:] = grad_y_ssc
                    llh_x_ssc = llh_y_ssc
            else:
                llh_x_ssc, _ = self.model_ssc.loss_and_grad(x, out=grad_x_ssc)

            rel_delta = relative_distance(x, prev_x)
            llh_x = self.model_ssc.original_loss(llh_x_ssc)
            obj = self.objective(x, loss=llh_x)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            obj_gain = prev_obj - obj

            converged = rel_obj < self.tol
            # if converged, we stop the loop and record the last step in history

            self._handle_history(n_iter, force=converged, obj=obj, x=x.copy(),
                                 rel_delta=rel_delta, step=alpha_k,
                                 rel_obj=rel_obj, l_k=l_k, beta_k=beta_k,
                                 lambda_k=lambda_k, th_gain=self._th_gain,
                                 obj_gain=obj_gain)
            if converged:
                break
        self._set("solution", x)
        return x

    def _handle_history(self, n_iter: int, force: bool = False, **kwargs):
        """Updates the history of the solver.
        """
        if n_iter == 0:
            self._set('_initial_n_hessiannorm_calls',
                      self.model.n_calls_hessian_norm)

        hessiannorm_calls = self.model.n_calls_hessian_norm - \
                            self._initial_n_hessiannorm_calls
        SolverFirstOrder._handle_history(self, n_iter, force=force,
                                         n_calls_hessiannorm=hessiannorm_calls,
                                         **kwargs)
