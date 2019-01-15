
"""
==============================
Asynchronous stochastic solver
==============================

This example illustrates the convergence speed of the asynchronous version of
SVRG and SAGA solvers. This solver respectively called KroMagnon and ASAGA
have been introduced in

* Mania, H., Pan, X., Papailiopoulos, D., Recht, B., Ramchandran, K. and Jordan, M.I., 2015.
  Perturbed iterate analysis for asynchronous stochastic optimization.
  `arXiv preprint arXiv:1507.06970.`_.

* R. Leblond, F. Pedregosa, and S. Lacoste-Julien: Asaga: Asynchronous
  Parallel Saga, `(AISTATS) 2017`_.

.. _arXiv preprint arXiv:1507.06970.: https://arxiv.org/abs/1507.06970
.. _(AISTATS) 2017: https://hal.inria.fr/hal-01665255/document

To obtain good speedup in a relative short time example we have designed very
sparse and ill-conditonned problem.
"""

from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from tick.linear_model import SimuLogReg, ModelLogReg
from tick.simulation import weights_sparse_gauss
from tick.solver import SVRG, SAGA
from tick.prox import ProxElasticNet, ProxL1

from tick.solver.build.solver import (
    SAGADouble as _SAGADouble,
    AtomicSAGADouble as _ASAGADouble,
    AtomicSAGADoubleAtomicIterate as _ASAGADoubleA,
    SAGADoubleAtomicIterate as _SAGADoubleA,
    AtomicSAGARelax as _ASAGADoubleRelax,
    AtomicSAGANoLoad as _ASAGADoubleNoLoad
)


seed = 1398
np.random.seed(seed)

def build_saga_solver(solver_class, model, prox, step, seed, n_threads):
    solver = SAGA(step=step, seed=seed, max_iter=100,
                  verbose=False, n_threads=n_threads, tol=0,
                  record_every=3)

    epoch_size = 0
    tol = solver.tol
    _rand_type = solver._rand_type
    step = solver.step
    record_every = solver.record_every
    seed = solver.seed
    n_threads = solver.n_threads

    solver._set('_solver',
                solver_class(epoch_size, tol, _rand_type, step,
                             record_every, seed, n_threads))

    if solver_class in [_SAGADoubleA, _ASAGADoubleA]:
        solver.set_model(model.to_atomic()).set_prox(prox.to_atomic())
    else:
        solver.set_model(model).set_prox(prox)

    return solver


n_samples = 50000
nnz = 20
sparsity = 1e-3
n_features = int(nnz / sparsity)
penalty_strength = 1e-5

weights = weights_sparse_gauss(n_features, nnz=1000)
intercept = 0.2
features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')

simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                       verbose=False, intercept=intercept)
features, labels = simulator.simulate()

model = ModelLogReg(fit_intercept=True)
model.fit(features, labels)
prox = ProxL1(penalty_strength, range=(0, n_features))
svrg_step = 1. / model.get_lip_max()

test_n_threads = [1, 2, 3, 4]

fig, axes = plt.subplots(1, 3, figsize=(10, 4))


classes = [_ASAGADoubleA, _ASAGADouble, _ASAGADoubleRelax]
class_names = ['Atomic $w$ and $\\alpha$', 'Atomic $\\alpha$', 'Atomic $\\alpha$ relax']

for ax, SolverClass, solver_name in zip(axes, classes, class_names):
    solver_list = []
    solver_labels = []

    for n_threads in test_n_threads:
        solver = build_saga_solver(SolverClass, model, prox, svrg_step, seed, n_threads)
        solver.solve()

        solver_list += [solver]

    solver_objectives = np.array([
        solver.history.values['obj'] for solver in solver_list])

    dist_solver_objectives = solver_objectives - solver_objectives.min()

    ax.plot([test_n_threads[0], test_n_threads[-1]], [1, test_n_threads[-1]],
            ls='--', lw=1, c='black')

    for target_precision in [1e-4, 1e-6, 1e-8, 1e-10]:
        target_indexes = [
            np.argwhere(dist_solver_objectives[i] < target_precision)[0][0]
            if dist_solver_objectives[i].min() < target_precision
            else np.nan
            for i in range(len(dist_solver_objectives))
        ]
        print(target_precision, target_indexes)

        target_times = np.array([
            solver.history.values['time'][index]
            if not np.isnan(index)
            else np.nan
            for index, solver in zip(target_indexes, solver_list)])

        time_one = target_times[0]
        y = time_one / target_times
        ax.plot(np.array(test_n_threads)[~np.isnan(target_times)],
                y[~np.isnan(target_times)], marker='x',
                label='{:.1g}'.format(target_precision))
        ax.set_xlabel('number of cores 2')
        ax.set_ylabel('speedup')
        ax.set_title(solver_name)

        ax.legend()

fig.tight_layout()
plt.show()
