# License: BSD 3 clause


from tick.optim.solver.base import SolverFirstOrderSto
from .build.solver import SDCA as _SDCA
import numpy as np


class SDCA(SolverFirstOrderSto):
    """Stochastic Dual Coordinate Ascent solver

    Parameters
    ----------
    l_l2sq : `float`
        Level of L2 penalization. L2 penalization is mandatory for SDCA.

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

    verbose : `bool`, default=`True`
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
    model : `Solver`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

    dual_solution : `np.ndarray`
        Dual vector to which the solver has converged
    """

    _attrinfos = {
        'l_l2sq': {'cpp_setter': 'set_l_l2sq'}
    }

    def __init__(self, l_l2sq: float, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1,
                 seed: int = -1):

        SolverFirstOrderSto.__init__(self, step=0, epoch_size=epoch_size,
                                     rand_type=rand_type, tol=tol,
                                     max_iter=max_iter, verbose=verbose,
                                     print_every=print_every,
                                     record_every=record_every, seed=seed)
        self.l_l2sq = l_l2sq
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0
        # Construct the wrapped C++ SDCA solver
        self._solver = _SDCA(self.l_l2sq, epoch_size,
                             self.tol, self._rand_type, self.seed)

    def objective(self, coeffs, loss: float = None):
        """Compute the objective minimized by the solver at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The objective is computed at this point

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(coeffs) ** 2
        return SolverFirstOrderSto.objective(self, coeffs, loss) + prox_l2_value

    def dual_objective(self, dual_coeffs):
        primal = self.model._sdca_primal_dual_relation(self.l_l2sq, dual_coeffs)
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(primal) ** 2
        return self.model.dual_loss(dual_coeffs) - prox_l2_value

    def _set_rand_max(self, model):
        try:
            # Some model, like Poisreg with linear link, have a special
            # rand_max for SDCA
            model_rand_max = model._sdca_rand_max
        except (AttributeError, NotImplementedError):
            model_rand_max = model._rand_max

        self._set("_rand_max", model_rand_max)

    @property
    def dual_solution(self):
        return self._solver.get_dual_vector()
