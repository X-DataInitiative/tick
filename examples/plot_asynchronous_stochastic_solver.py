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

from tick.dataset import fetch_tick_dataset
from tick.plot import plot_history
import numpy as np
from tick.linear_model import SimuLogReg, ModelLogReg

from tick.simulation import weights_sparse_gauss
from tick.solver import SVRG, SAGA
from tick.prox import ProxElasticNet, ProxL1

from tick.linear_model.build.linear_model import ModelLogRegAtomicDouble
from tick.prox.build.prox import ProxL1AtomicDouble

from tick.solver.build.solver import (
    SAGADouble as _SAGADouble,
    AtomicSAGADouble as _ASAGADouble,
    AtomicSAGADoubleAtomicIterate as _ASAGADoubleA,
    SAGADoubleAtomicIterate as _SAGADoubleA,
    AtomicSAGARelax as _ASAGADoubleRelax,
)

seed = 1398
np.random.seed(seed)

n_samples = 40000
nnz = 50
sparsity = 1e-2
n_features = int(nnz / sparsity)
penalty_strength = 1e-5


simulate = True
if simulate:
    weights = weights_sparse_gauss(n_features, nnz=1000)
    intercept = 0.2
    features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')

    simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                           verbose=False, intercept=intercept)
    features, labels = simulator.simulate()

else:
    features, labels = fetch_tick_dataset("binary/kdd2010/kdd2010.trn.bz2")
    print(features.shape)

model = ModelLogReg(fit_intercept=False)
model.fit(features, labels)
prox = ProxL1(penalty_strength)
svrg_step = 1. / model.get_lip_max()

test_n_threads = [1, 2, 4, 8]
threads_ls = {1: '-', 2: '--', 4: ':', 8: '-'}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

classes = [_SAGADouble, _SAGADoubleA, _ASAGADouble, _ASAGADoubleA, _ASAGADoubleRelax]
class_names = ['Wild', 'Atomic $w$', 'Atomic $\\alpha$', 'Atomic $w$ and $\\alpha$',
               'Atomic $\\alpha$ relax']

for solver_class, solver_name in zip(classes, class_names):  # [SVRG, SAGA]):
    solver_list = []
    solver_labels = []

    for n_threads in test_n_threads:
        solver = SAGA(step=svrg_step, seed=seed, max_iter=100,
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

        solver.solve()

        solver_list += [solver]
        solver_labels += ['{} {}'.format(solver_name, n_threads)]

    plot_history(solver_list, dist_min=True, log_scale=True,
                 labels=solver_labels, ax=ax, x='time')

    for j, line in enumerate(ax.lines[-len(test_n_threads):]):
        print(j, solver_name, test_n_threads[j])
        line.set_color('C{}'.format(class_names.index(solver_name)))
        line.set_linestyle(threads_ls[test_n_threads[j]])

    ax.set_ylabel('log distance to optimal objective', fontsize=14)

fig.tight_layout()
ax.legend()
ax.set_ylim([1e-10, None])
plt.show()
