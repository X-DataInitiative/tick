# License: BSD 3 clause

import numpy as np

from . import Solver
from tick.base_model import Model
from tick.prox.base import Prox

__author__ = 'Stephane Gaiffas'


class SolverFirstOrder(Solver):
    """The base class for a first order solver. It defines methods for
    setting a model (giving first order information) and a proximal
    operator

    In only deals with verbosing information, and setting parameters

    Parameters
    ----------
    step : `float` default=None
        Step-size of the algorithm

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

    Attributes
    ----------
    model : `Model`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "model": {
            "writable": False
        },
        "prox": {
            "writable": False
        },
        "_initial_n_calls_loss_and_grad": {
            "writable": False
        },
        "_initial_n_calls_loss": {
            "writable": False
        },
        "_initial_n_calls_grad": {
            "writable": False
        },
        "_initial_n_passes_over_data": {
            "writable": False
        },
    }

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):

        self.dtype = None

        Solver.__init__(self, tol, max_iter, verbose, print_every,
                        record_every)
        self.model = None
        self.prox = None
        self.step = step
        # Martin's complicated and useless stuff :)
        self._initial_n_calls_loss_and_grad = 0
        self._initial_n_calls_loss = 0
        self._initial_n_calls_grad = 0
        self._initial_n_passes_over_data = 0

    def validate_model(self, model: Model):
        if not isinstance(model, Model):
            raise ValueError('Passed object of class %s is not a '
                             'Model class' % model.name)
        if not model._fitted:
            raise ValueError('Passed object %s has not been fitted. You must '
                             'call ``fit`` on it before passing it to '
                             '``set_model``' % model.name)

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
            The same instance with given model
        """
        self.validate_model(model)
        self.dtype = model.dtype
        self._set("model", model)
        return self

    def _initialize_values(self, x0: np.ndarray = None, step: float = None,
                           n_empty_vectors: int = 0):
        """Initialize values

        Parameters
        ----------
        x0 : `numpy.ndarray`
            Starting point

        step : `float`
            Initial step

        n_empty_vectors : `int`
            Number of empty vector of like x0 needed

        Returns
        -------
        step : `float`
            Initial step

        obj : `float`
            Initial value of objective function

        iterate : `numpy.ndarray`
            copy of starting point

        empty vectors : `numpy.ndarray`
            n_empty_vectors empty vectors shaped as x0. For example, those
            vectors can be used to store previous iterate values during
            a solver execution.
        """
        # Initialization
        if step is None:
            if self.step is None:
                raise ValueError("No step specified.")
            else:
                step = self.step
        else:
            self.step = step
        if x0 is None:
            x0 = np.zeros(self.model.n_coeffs, dtype=self.dtype)
        iterate = x0.copy()
        obj = self.objective(iterate)

        result = [step, obj, iterate]
        for _ in range(n_empty_vectors):
            result.append(np.zeros_like(x0))

        return tuple(result)

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
        if not isinstance(prox, Prox):
            raise ValueError('Passed object of class %s is not a '
                             'Prox class' % prox.name)
        if self.dtype is None or self.model is None:
            raise ValueError("Solver must call set_model before set_prox")
        if prox.dtype != self.dtype:
            prox = prox.astype(self.dtype)
        self._set("prox", prox)
        return self

    def astype(self, dtype_or_object_with_dtype):
        if self.model is None:
            raise ValueError("Cannot reassign solver without a model")

        import tick.base.dtype_to_cpp_type
        new_solver = tick.base.dtype_to_cpp_type.copy_with(
            self,
            ["prox", "model"]  # ignore on deepcopy
        )
        new_solver.dtype = tick.base.dtype_to_cpp_type.extract_dtype(
            dtype_or_object_with_dtype)
        new_solver.set_model(self.model.astype(new_solver.dtype))
        if self.prox is not None:
            new_solver.set_prox(self.prox.astype(new_solver.dtype))
        return new_solver

    def _as_dict(self):
        dd = Solver._as_dict(self)
        if self.model is not None:
            dd["model"] = self.model._as_dict()
        if self.prox is not None:
            dd["prox"] = self.prox._as_dict()
        return dd

    def objective(self, coeffs, loss: float = None):
        """Compute the objective function

        Parameters
        ----------
        coeffs : `np.array`, shape=(n_coeffs,)
            Point where the objective is computed

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """
        if self.prox is None:
            prox_value = 0
        else:
            prox_value = self.prox.value(coeffs)

        if loss is None:
            return self.model.loss(coeffs) + prox_value
        else:
            return loss + prox_value

    def solve(self, x0=None, step=None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : `np.array`, shape=(n_coeffs,), default=`None`
            Starting point of the solver

        step : `float`, default=`None`
            Step-size or learning rate for the solver. This can be tuned also
            using the ``step`` attribute

        Returns
        -------
        output : `np.array`, shape=(n_coeffs,)
            Obtained minimizer for the problem, same as ``solution`` attribute
        """
        if x0 is not None and self.dtype is not "float64":
            x0 = x0.astype(self.dtype)

        if self.model is None:
            raise ValueError('You must first set the model using '
                             '``set_model``.')
        if self.prox is None:
            raise ValueError('You must first set the prox using '
                             '``set_prox``.')
        solution = Solver.solve(self, x0, step)
        return solution

    def _handle_history(self, n_iter: int, force: bool = False, **kwargs):
        """Updates the history of the solver.

        Parameters
        ----------

        Notes
        -----
        This should not be used by end-users.
        """
        # self.model.n_calls_loss_and_grad is shared by all
        # solvers using this model
        # hence it might not be at 0 while starting
        # /!\ beware if parallel computing...
        if n_iter == 1:
            self._set("_initial_n_calls_loss_and_grad",
                      self.model.n_calls_loss_and_grad)
            self._set("_initial_n_calls_loss", self.model.n_calls_loss)
            self._set("_initial_n_calls_grad", self.model.n_calls_grad)
            self._set("_initial_n_passes_over_data",
                      self.model.n_passes_over_data)
        n_calls_loss_and_grad = \
            self.model.n_calls_loss_and_grad - \
            self._initial_n_calls_loss_and_grad
        n_calls_loss = \
            self.model.n_calls_loss - self._initial_n_calls_loss
        n_calls_grad = \
            self.model.n_calls_grad - self._initial_n_calls_grad
        n_passes_over_data = \
            self.model.n_passes_over_data - \
            self._initial_n_passes_over_data
        Solver.\
            _handle_history(self, n_iter, force=force,
                            n_calls_loss_and_grad=n_calls_loss_and_grad,
                            n_calls_loss=n_calls_loss,
                            n_calls_grad=n_calls_grad,
                            n_passes_over_data=n_passes_over_data,
                            **kwargs)
