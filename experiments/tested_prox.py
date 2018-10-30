import traceback
from abc import abstractmethod

import numpy as np

from experiments.hawkes_coeffs import dim_from_n
from tick.plot import plot_history
from tick.prox import ProxL1, ProxL1w, ProxNuclear
from tick.hawkes import ModelHawkesExpKernLeastSq
from tick.prox.base import Prox


class ProxTensorNuclear(Prox):
    """Prox nuclear applied on 3d tensor
    Designed for adjacency tensor of HawkesSumExp
    """
    _attrinfos = {
        '_nuclear_prox': {'writable': False},
    }

    def __init__(self, strength: float, n_rows: int = None,
                 depth: int = None, range: tuple = None,
                 positive: bool = False, logger=None):
        Prox.__init__(self, range=range)
        self.strength = strength
        self.n_rows = n_rows
        self.positive = positive
        self.depth = depth
        self._nuclear_prox = None
        if logger is None:
            self.logger = print
        else:
            self.logger = logger

    def _get_tensor(self, coeffs):
        if self.n_rows is None or self.depth is None:
            raise ValueError(
                "'n_rows and depth parameter must be set before, either "
                "in constructor or manually")

        if self.range is None:
            start, end = 0, coeffs.shape[0]
        else:
            start, end = self.range
        if (end - start) % (self.n_rows * self.depth) != 0:
            raise ValueError("``end``-``start`` must be a multiple of "
                             "``n_rows` * `depth``")

        n_cols = int((end - start) / (self.n_rows * self.depth))
        x = coeffs[start:end].copy().reshape((self.n_rows, n_cols, self.depth))
        return x

    @abstractmethod
    def _get_stacked_matrix(self, coeffs):
        pass

    @abstractmethod
    def _get_flat_tensor_from_stacked(self, hstack_coeffs):
        pass

    def _call(self, coeffs: np.ndarray, step: float, out: np.ndarray):
        stacked_coeffs = self._get_stacked_matrix(coeffs).ravel()
        out_stacked_coeffs = np.zeros_like(stacked_coeffs)
        try:
            self._nuclear_prox._call(stacked_coeffs, step, out=out_stacked_coeffs)
        except np.linalg.linalg.LinAlgError:
            self.logger('Failed convergence in call')
            self.logger(stacked_coeffs.min(), stacked_coeffs.max())
            self.logger(stacked_coeffs)
            traceback.print_exc()
            out_stacked_coeffs = stacked_coeffs

        stacked_delta = out_stacked_coeffs - stacked_coeffs
        flat_delta = self._get_flat_tensor_from_stacked(stacked_delta)

        if self.range is None:
            start, end = 0, coeffs.shape[0]
        else:
            start, end = self.range
        out[start:end] = coeffs[start:end] + flat_delta

    def value(self, coeffs: np.ndarray) -> float:
        stacked_coeffs = self._get_stacked_matrix(coeffs).ravel()
        try:
            return self._nuclear_prox.value(stacked_coeffs)
        except np.linalg.linalg.LinAlgError:
            self.logger('Failed convergence in value')
            self.logger(stacked_coeffs.min(), stacked_coeffs.max())
            self.logger(stacked_coeffs)
            traceback.print_exc()
            return 0


class ProxTensorHStackNuclear(ProxTensorNuclear):
    def __init__(self, strength: float, n_rows: int = None,
                 depth: int = None, range: tuple = None,
                 positive: bool = False, logger=None):
        ProxTensorNuclear.__init__(self, strength, n_rows=n_rows,
                                   depth=depth, range=range, positive=positive,
                                   logger=logger)
        self._nuclear_prox = ProxNuclear(strength, n_rows=n_rows * depth)

    def _get_stacked_matrix(self, coeffs):
        tensor = self._get_tensor(coeffs)
        return np.hstack((tensor[:, :, k] for k in range(self.depth)))

    def _get_flat_tensor_from_stacked(self, hstack_coeffs):
        hstack_matrix = hstack_coeffs.reshape(self.n_rows, -1)
        flat_tensor = np.zeros(np.prod(hstack_matrix.shape))
        for u in range(self.depth):
            first_col = u * self.n_rows
            last_col = (u + 1) * self.n_rows
            flat_tensor[u::self.depth] = \
                hstack_matrix[:, first_col:last_col].ravel()

        return flat_tensor


class ProxTensorVStackNuclear(ProxTensorNuclear):
    def __init__(self, strength: float, n_rows: int = None,
                 depth: int = None, range: tuple = None,
                 positive: bool = False, logger=None):
        ProxTensorNuclear.__init__(self, strength, n_rows=n_rows,
                                   depth=depth, range=range, positive=positive,
                                   logger=logger)
        self._nuclear_prox = ProxNuclear(strength, n_rows=n_rows)

    def _get_stacked_matrix(self, coeffs):
        tensor = self._get_tensor(coeffs)
        return np.vstack((tensor[:, :, k] for k in range(self.depth)))

    def _get_flat_tensor_from_stacked(self, vstack_coeffs):
        vstack_matrix = vstack_coeffs.reshape(self.depth * self.n_rows, -1)
        flat_tensor = np.zeros(np.prod(vstack_matrix.shape))
        for u in range(self.depth):
            first_row = u * self.n_rows
            last_row = (u + 1) * self.n_rows
            flat_tensor[u::self.depth] = \
                vstack_matrix[first_row:last_row, :].ravel()

        return flat_tensor


def get_n_decays_from_model(model):
    if isinstance(model, ModelHawkesExpKernLeastSq):
        return 1
    else:
        return model.n_decays


def create_prox_l1_no_mu(strength, model, logger=None):
    dim = dim_from_n(model.n_coeffs, get_n_decays_from_model(model))
    prox = ProxL1(strength, positive=True, range=(dim, dim * dim + dim))
    return prox


def create_prox_l1w_no_mu(strength, model, logger=None):
    weights = model.compute_penalization_constant(strength=strength)
    n_decays = get_n_decays_from_model(model)
    dim = dim_from_n(model.n_coeffs, n_decays)
    weights = weights[dim:]
    prox = ProxL1w(1, positive=True, weights=weights,
                   range=(dim, dim * dim * n_decays + dim))
    return prox


def create_prox_l1w_no_mu_un(strength, model, logger=None):
    weights = model.compute_penalization_constant()
    n_decays = get_n_decays_from_model(model)
    dim = dim_from_n(model.n_coeffs, n_decays)
    weights = weights[dim:]
    prox = ProxL1w(strength, positive=True, weights=weights,
                   range=(dim, dim * dim * n_decays + dim))
    return prox

# Nuclear
def create_prox_nuclear(strength, model, logger=None):
    n_decays = get_n_decays_from_model(model)
    dim = dim_from_n(model.n_coeffs, n_decays)

    if n_decays == 1:
        prox_range = (dim, dim * dim + dim)
        n_rows = dim
        prox = ProxNuclear(strength, n_rows, range=prox_range, positive=True)
        return prox,
    else:
        prox_range = (dim, dim * dim * n_decays + dim)
        n_rows = dim
        prox_hstack = ProxTensorHStackNuclear(strength, n_rows, n_decays,
                                              range=prox_range, positive=True,
                                              logger=logger)
        prox_vstack = ProxTensorVStackNuclear(strength, n_rows, n_decays,
                                              range=prox_range, positive=True)
        return prox_hstack, prox_vstack


def create_prox_l1w_no_mu_nuclear(strength, model, logger=None):
    l1, tau = strength
    prox_l1 = create_prox_l1w_no_mu(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model, logger=logger)
    return (prox_l1,) + prox_nuclear


def create_prox_l1w_un_no_mu_nuclear(strength, model, logger=None):
    l1, tau = strength
    prox_l1 = create_prox_l1w_no_mu_un(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model, logger=logger)
    return (prox_l1,) + prox_nuclear


def create_prox_l1_no_mu_nuclear(strength, model, logger=None):
    l1, tau = strength
    prox_l1 = create_prox_l1_no_mu(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model, logger=logger)
    return (prox_l1,) + prox_nuclear


if __name__ == '__main__':
    from tick.hawkes import ModelHawkesSumExpKernLeastSq
    from tick.solver import AGD, GD, GFB

    n_nodes_ = 3
    n_decays_ = 2
    fake_tensor_ = np.arange(n_nodes_ * n_nodes_ * n_decays_) \
        .reshape(n_nodes_, n_nodes_, n_decays_).astype(float)

    model_ = ModelHawkesSumExpKernLeastSq(np.random.rand(n_decays_))
    model_.fit([np.cumsum(np.random.rand(100)) for _ in range(n_nodes_)])

    for u in range(n_decays_):
        print(fake_tensor_[:, :, u])

    prox_hstack_, prox_vstack_ = create_prox_nuclear(.01, model_)
    fake_coeffs_ = np.hstack((np.zeros(n_nodes_), fake_tensor_.ravel()))

    np.testing.assert_array_equal(fake_tensor_,
                                  prox_hstack_._get_tensor(fake_coeffs_))
    print('prox._get_hstack_matrix')
    hstack_matrix_ = prox_hstack_._get_stacked_matrix(fake_coeffs_)
    print(hstack_matrix_)
    print('prox._get_vstack_matrix')
    vstack_matrix_ = prox_vstack_._get_stacked_matrix(fake_coeffs_)
    print(vstack_matrix_)

    np.testing.assert_array_equal(
        fake_tensor_.ravel(),
        prox_hstack_._get_flat_tensor_from_stacked(hstack_matrix_.ravel()))

    np.testing.assert_array_equal(
        fake_tensor_.ravel(),
        prox_vstack_._get_flat_tensor_from_stacked(vstack_matrix_.ravel()))

    for _ in range(5):
        print(prox_hstack_.value(fake_coeffs_))
        prox_hstack_.call(fake_coeffs_, out=fake_coeffs_)

    # solvers_ = []
    # # steps = [0.1, 0.5, 0.8, 1., 1.5, 2.]
    # steps = [1]
    # for coeff in steps:
    #     solver_ = GD(step=coeff / model_.get_lip_best(), linesearch=False,
    #                   print_every=200, max_iter=10000, tol=1e-10)
    #     solver_.set_model(model_).set_prox(prox_hstack_)
    #     solver_.solve(np.ones(model_.n_coeffs))
    #     solvers_ += [solver_]
    #
    # print('solver_.solution', solver_.solution)
    # print(solver_.objective(solver_.solution))
    #
    # plot_history(solvers_, labels=steps, log_scale=True, dist_min=True)
    #
    # w_ = solver_.solution
    # grad_ = w_ - solver_.prox.call(w_ - solver_.model.grad(w_))
    # print("grad_", grad_)
    #

    solvers_ = []
    # steps = [0.1, 0.5, 0.8, 1., 1.5, 2.]
    steps = [1]
    for coeff in steps:
        solver_ = GFB(step=coeff / model_.get_lip_best(),
                      print_every=200, max_iter=10000, tol=1e-10)
        solver_.set_model(model_).set_prox((prox_hstack_, prox_vstack_))
        solver_.solve(np.ones(model_.n_coeffs))
        solvers_ += [solver_]

    print('solver_.solution', solver_.solution)
    print(solver_.objective(solver_.solution))

    plot_history(solvers_, labels=steps, log_scale=True, dist_min=True)

    w_ = solver_.solution
    # grad_ = w_ - solver_.prox.call(w_ - solver_.model.grad(w_))
    # print("grad_", grad_)

