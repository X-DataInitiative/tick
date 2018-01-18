# License: BSD 3 clause

from tick.base_model import ModelGeneralizedLinear
from .base import SolverFirstOrderSto
from .build.solver import SAGA as _SAGA

__author__ = "Stephane Gaiffas"

variance_reduction_methods_mapper = {
    'last': _SAGA.VarianceReductionMethod_Last,
    'avg': _SAGA.VarianceReductionMethod_Average,
    'rand': _SAGA.VarianceReductionMethod_Random
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
    are updated at the beginning of each epoch, with a strategy that depend on
    the ``variance_reduction`` parameter. The step-size :math:`\\eta` can be
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
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    variance_reduction : {'last', 'avg', 'rand'}, default='last'
        Strategy used for the computation of the iterate used in
        variance reduction (also called phase iterate). A warning will be
        raised if the ``'avg'`` strategy is used when the model is a
        generalized linear model with sparse features, since it is strongly
        sub-optimal in this case

        * ``'last'`` : the phase iterate is the last iterate of the previous
          epoch
        * ``'avg``' : the phase iterate is the average over the iterates in the
          past epoch
        * ``'rand'``: the phase iterate is a random iterate of the previous
          epoch

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
    * A. Defazio, F. Bach, S. Lacoste-Julien, SAGA: A fast incremental gradient
      method with support for non-strongly convex composite objectives,
      NIPS 2014
    """

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1,
                 seed: int = -1, variance_reduction: str = "last"):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type,
                                     tol, max_iter, verbose,
                                     print_every, record_every, seed=seed)
        step = self.step
        if step is None:
            step = 0.

        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        # Construct the wrapped C++ SAGA solver
        self._solver = _SAGA(epoch_size, self.tol,
                             self._rand_type, step, self.seed)

        self.variance_reduction = variance_reduction

    @property
    def variance_reduction(self):
        return next((k for k, v in variance_reduction_methods_mapper.items()
                     if v == self._solver.get_variance_reduction()), None)

    @variance_reduction.setter
    def variance_reduction(self, val: str):

        if val not in variance_reduction_methods_mapper:
            raise ValueError(
                'variance_reduction should be one of "{}", got "{}".'.format(
                    ', '.join(variance_reduction_methods_mapper.keys()),
                    val))

        self._solver.set_variance_reduction(
            variance_reduction_methods_mapper[val])

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
        if isinstance(model, ModelGeneralizedLinear):
            return SolverFirstOrderSto.set_model(self, model)
        else:
            raise ValueError("SAGA accepts only childs of "
                             "`ModelGeneralizedLinear`")
