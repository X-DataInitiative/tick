# License: BSD 3 clause

import numpy as np

from .base import SolverFirstOrderSto
from .build.solver import SGDDouble as _SGDDouble
from .build.solver import SGDFloat as _SGDFloat

__author__ = "Stephane Gaiffas"

dtype_class_mapper = {
    np.dtype('float32'): _SGDFloat,
    np.dtype('float64'): _SGDDouble
}

# TODO: preparer methodes pour set et get attributes


class SGD(SolverFirstOrderSto):
    """Stochastic gradient descent solver

    For the minimization of objectives of the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds
    to the ``model.loss`` method of the model (passed with ``set_model`` to the
    solver) and :math:`g` corresponds to the ``prox.value`` method of the
    prox (passed with the ``set_prox`` method).
    One iteration of :class:`SGD <tick.solver.SGD>` corresponds to the
    following iteration applied ``epoch_size`` times:

    .. math::
        w^{t+1} \\gets \\mathrm{prox}_{\\eta_t g} \\big(w^t - \\eta_t
        \\nabla f_i(w^t) \\big),

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration, where :math:`\\eta_t = \eta / (t + 1)`, with
    :math:`\\eta > 0` that can be tuned with ``step``. The seed of the random
    number generator for generation of samples :math:`i` can be seeded with
    ``seed``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` epochs (namely ``max_iter``:math:`\\times` ``epoch_size``
    iterations).
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver.

    Parameters
    ----------
    step : `float`
        Step-size parameter, the most important parameter of the solver.
        A try-an-improve approach should be used.

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver, namely maximum number of
        epochs (by default full pass over the data, unless ``epoch_size`` has
        been modified from default)

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    seed : `int`, default=-1
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    epoch_size : `int`, default given by model
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

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
    * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    """

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 1e-10,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1, seed: int = -1):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type, tol,
                                     max_iter, verbose, print_every,
                                     record_every, seed)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                             dtype_class_mapper)

        # Type mapping None to unsigned long and double does not work...
        step = self.step
        if step is None:
            step = 0.
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        self._set(
            '_solver',
            solver_class(epoch_size, self.tol, self._rand_type, step,
                         self.record_every, self.seed))
