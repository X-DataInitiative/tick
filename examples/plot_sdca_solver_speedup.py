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
from tick.linear_model import ModelLinReg, SimuLinReg, SimuLogReg, ModelLogReg, ModelPoisReg, SimuPoisReg
from tick.plot import plot_history

from tick.simulation import weights_sparse_gauss
from tick.solver import SDCA
from tick.solver.sdca import AtomicSDCA

from tick.prox import ProxZero, ProxL1
from tick.prox.prox_zero import AtomicProxZero

seed = 1398
np.random.seed(seed)

n_samples = 20000
n_features = 50000
sparsity = 1e-3
penalty_strength = 1e-3

weights = weights_sparse_gauss(n_features, nnz=500)
intercept = 0.2
features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')
print(features.shape, len(features.nonzero()[0]))

SimuClass = SimuLogReg  # SimuLinReg  # SimuPoisReg  #
ModelClass = ModelLogReg  # ModelLinReg  # ModelPoisReg  #

simulator = SimuClass(weights, n_samples=n_samples, features=features,
                       verbose=False, intercept=intercept, dtype=float)
features, labels = simulator.simulate()

model = ModelClass(fit_intercept=True)
model.fit(features, labels)


prox = ProxL1(1e-4)
atomic_prox = prox.to_atomic()

sdca = SDCA(penalty_strength, tol=0, verbose=False)

sdca.set_model(model)
sdca.set_prox(prox)

atomic_sdca = AtomicSDCA(penalty_strength, tol=0, verbose=False, n_threads=3)
atomic_sdca.set_model(model).set_prox(atomic_prox, prox)

sdca.solve()
sdca.print_history()

atomic_sdca.solve()
atomic_sdca.print_history()

batch_sizes = [1, 3, 10]
test_n_threads = [1, 2, 3, 4]#, 3, 4, 8]

fig, axes = plt.subplots(2, len(batch_sizes), figsize=(3 * len(batch_sizes), 6))


for i, batch_size in enumerate(batch_sizes):
    axes[0, i].set_title('batch {}'.format(batch_size))
    solver_list = []
    solver_labels = []

    for n_threads in test_n_threads:

        if n_threads == 1 and False:
            solver = SDCA(penalty_strength, tol=0, verbose=False,
                          seed=seed, max_iter=100)
            solver.set_model(model).set_prox(prox)
            solver_labels += [solver.name]
        else:
            solver = AtomicSDCA(penalty_strength, tol=0, verbose=False,
                                seed=seed, max_iter=100, n_threads=n_threads,
                                batch_size=batch_size)
            solver.set_model(model).set_prox(atomic_prox, prox)
            solver_labels += ['{} {}'.format(solver.name, n_threads)]

        solver.solve()
        solver_list += [solver]

    plot_history(solver_list, x="time", dist_min=True, log_scale=True,
                 labels=solver_labels, ax=axes[0, i])

    solver_objectives = np.array([
        solver.history.values['obj'] for solver in solver_list])

    dist_solver_objectives = solver_objectives - solver_objectives.min()

    # speedup
    ax = axes[1, i]
    ax.plot([test_n_threads[0], test_n_threads[-1]], [1, test_n_threads[-1]],
            ls='--', lw=1, c='black')

    for target_precision in [1e-4, 1e-6, 1e-8, 1e-10]:
        target_indexes = [
            np.argwhere(dist_solver_objectives[i] < target_precision)[0][0]
            if dist_solver_objectives[i].min() < target_precision
            else np.nan
            for i in range(len(dist_solver_objectives))]
        print(target_precision, target_indexes)

        target_times = np.array([
            solver.history.values['time'][index]
            if not np.isnan(index)
            else np.nan
            for index, solver in zip(target_indexes, solver_list)])

        time_one = target_times[0]
        y = time_one / target_times
        ax.plot(test_n_threads, y, marker='x', label='{:.1g}'
                .format(target_precision))
        ax.set_xlabel('number of cores')
        ax.set_ylabel('speedup')
        ax.set_title(solver_list[0].name)

        ax.legend()

fig.tight_layout()
plt.show()
