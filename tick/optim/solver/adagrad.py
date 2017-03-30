from tick.optim.solver.base import SolverFirstOrderSto
from tick.optim.solver.build.solver import AdaGrad as _AdaGrad

__author__ = "Søren Vinther Poulsen"


class AdaGrad(SolverFirstOrderSto):
    """
    Adaptive Gradient Descent solver

    Based on the works by:

    Duchi, J., Hazan, E., & Singer, Y. (2011).
    Adaptive Subgradient Methods for Online Learning and
    Stochastic Optimization. Journal of Machine Learning Research.

    Parameters
    ----------
    step : `float` default=0.01
        Step-size of the algorithm

    epoch_size : `int`
        Epoch size

    rand_type : `str`
        Type of random sampling

        * if ``"unif"`` samples are uniformly drawn among all possibilities
        * if ``"perm"`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

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

    seed : `int`
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    Attributes
    ----------
    model : `Solver`
        The model to solve

    prox : `Prox`
        Proximal operator to solve
    """

    def __init__(self, step: float = 0.01, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1,
                 seed: int = -1):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type,
                                     tol, max_iter, verbose,
                                     print_every, record_every, seed)
        # Type mapping None to unsigned long and double does not work...
        step = self.step
        if step is None:
            step = 0.
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        # Construct the wrapped C++ AdaGrad solver
        self._solver = _AdaGrad(epoch_size, self.tol, self._rand_type, step,
                                self.seed)
