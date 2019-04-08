# License: BSD 3 clause

__author__ = "Stephane Gaiffas"

import numpy as np

from tick.base_model import ModelGeneralizedLinear
from tick.solver.base import SolverFirstOrderSto, SolverSto

from tick.solver.build.solver import SAGADouble as _SAGADouble
from tick.solver.build.solver import SAGAFloat as _SAGAFloat
dtype_class_mapper = {
    np.dtype('float32'): _SAGAFloat,
    np.dtype('float64'): _SAGADouble
}

from tick.solver.build.solver import AtomicSAGADouble as _ASAGADouble
from tick.solver.build.solver import AtomicSAGAFloat as _ASAGAFloat
dtype_atomic_mapper = {
    np.dtype('float32'): _ASAGAFloat,
    np.dtype('float64'): _ASAGADouble
}


class SAGA(SolverFirstOrderSto):
    """Stochastic Average Gradient solver, for the minimization of objectives
    of the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. Note that :class:`SAGA <tick.solver.SAGA>` works only
    with linear models, see :ref:`linear_model` and :ref:`robust`, where all
    linear models are listed.
    Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds
    to the ``model.loss`` method of the model (passed with ``set_model`` to the
    solver) and :math:`g` corresponds to the ``prox.value`` method of the
    prox (passed with the ``set_prox`` method).
    One iteration of :class:`SAGA <tick.solver.SAGA>` corresponds to the
    following iteration applied ``epoch_size`` times:

    .. math::
        \\begin{align*}
        w &\\gets \\mathrm{prox}_{\\eta g} \\Big(w - \\eta \\Big(\\nabla f_i(w)
         - \\delta_i + \\frac 1n \\sum_{i'=1}^n \\delta_{i'} \\Big) \\Big) \\\\
        \\delta_i &\\gets \\nabla f_i(w)
        \\end{align*}

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration, and where :math:`\\bar w` and :math:`\\nabla f(\\bar w)`
    are updated at the beginning of each epoch. The step-size :math:`\\eta` can be
    tuned with ``step``, the seed of the random number generator for generation
    of samples :math:`i` can be seeded with ``seed``. The iterations stop
    whenever tolerance ``tol`` is achieved, or after ``max_iter`` epochs
    (namely ``max_iter`` :math:`\\times` ``epoch_size`` iterates).
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver.

    Internally, :class:`SAGA <tick.solver.SAGA>` has dedicated code when
    the model is a generalized linear model with sparse features, and a
    separable proximal operator: in this case, each iteration works only in the
    set of non-zero features, leading to much faster iterates.

    Parameters
    ----------
    step : `float`
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_max()``. Otherwise, use a try-an-improve
        approach

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=10
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
        Epoch size, by default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    print_every : `int`, default=1
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    n_threads : `int`, default=1
        Number of threads to use for parallel optimization. The strategy used
        for this is asynchronous updates of the iterates.

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
    * A. Defazio, F. Bach, S. Lacoste-Julien, SAGA: A fast incremental gradient
      method with support for non-strongly convex composite objectives,
      NIPS 2014

    * R. Leblond, F. Pedregosa, and S. Lacoste-Julien: Asaga: Asynchronous
      Parallel Saga, (AISTATS) 2017
    """
    _attrinfos = {"n_threads": {"writable": False}}

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 0., max_iter: int = 100,
                 verbose: bool = True, print_every: int = 10,
                 record_every: int = 1, seed: int = -1, n_threads: int = 1):
        self.n_threads = n_threads

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type, tol,
                                     max_iter, verbose, print_every,
                                     record_every, seed=seed)

    def set_model(self, model: ModelGeneralizedLinear):
        """Set model in the solver

        Parameters
        ----------
        model : `ModelGeneralizedLinear`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things). SAGA only accepts childs of `ModelGeneralizedLinear`

        Returns
        -------
        output : `Solver`
            The `Solver` with given model
        """
        if not isinstance(model, ModelGeneralizedLinear):
            raise ValueError("SAGA accepts only childs of "
                             "`ModelGeneralizedLinear`")

        if hasattr(model, "n_threads"):
            model.n_threads = self.n_threads

        return SolverFirstOrderSto.set_model(self, model)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        # Construct the wrapped C++ SAGA solver
        step = self.step
        if step is None:
            step = 0.
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        if self.n_threads == 1:
            solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                                 dtype_class_mapper)
            self._set(
                '_solver',
                solver_class(epoch_size, self.tol, self._rand_type, step,
                             self.record_every, self.seed))
        else:
            solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                                 dtype_atomic_mapper)
            self._set(
                '_solver',
                solver_class(epoch_size, self.tol, self._rand_type, step,
                             self.record_every, self.seed, self.n_threads))
