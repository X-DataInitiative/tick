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

import sys
from scipy import sparse
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from tick.plot import plot_history
import numpy as np
import pandas as pd
import seaborn as sns
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
    AtomicSAGANoLoad as _ASAGADoubleNoLoad
)


def build_saga_solver(solver_class, model, prox, step, seed, n_threads):
    solver = SAGA(step=step, seed=seed, max_iter=50,
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


seed = 1398
np.random.seed(seed)

n_samples = 40000
nnz = 20
sparsity = 1e-1
n_features = int(nnz / sparsity)
penalty_strength = 1e-5

weights = weights_sparse_gauss(n_features, nnz=1000)
intercept = 0.2
features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')

simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                       verbose=False, intercept=intercept)
features, labels = simulator.simulate()

model = ModelLogReg(fit_intercept=False)
model.fit(features, labels)
prox = ProxL1(penalty_strength)
svrg_step = 1. / model.get_lip_max()

n_threads = int(sys.argv[1])

print('\n FOR {} THREADS'.format(n_threads))


classes = [_SAGADouble, _SAGADoubleA, _ASAGADouble, _ASAGADoubleA]
class_names = ['Wild', 'Atomic $w$', 'Atomic $\\alpha$', 'Atomic $w$ and $\\alpha$']


df_infos = []
for _ in range(3):
    solvers = []
    solver_names = []

    solvers += [build_saga_solver(_SAGADouble, model, prox, svrg_step, seed, n_threads)]
    solver_names += ['Wild']

    solvers += [build_saga_solver(_ASAGADoubleA, model, prox, svrg_step, seed, n_threads)]
    solver_names += ['Atomic $w$ and $\\alpha$']

    solver = build_saga_solver(_ASAGADouble, model, prox, svrg_step, seed, n_threads)
    solvers += [solver]
    solver_names += ['Atomic $\\alpha$']

    solver = build_saga_solver(_ASAGADoubleRelax, model, prox, svrg_step, seed, n_threads)
    solvers += [solver]
    solver_names += ['Atomic $\\alpha$, relax']

    solver = build_saga_solver(_ASAGADoubleNoLoad, model, prox, svrg_step, seed, n_threads)
    solvers += [solver]
    solver_names += ['Atomic $\\alpha$, relax, no load']

    for solver, solver_name in zip(solvers, solver_names):
        solver.solve()
        print(solver_name.ljust(35), _, solver.history.last_values['time'])
        df_infos += [{'name': solver_name, 'time': solver.history.last_values['time']}]
    print('-' * 45)

df = pd.DataFrame(df_infos)

sns.barplot(x="name", y="time", data=df, ci='sd')

plt.show()
# df.to_csv('timings_{}_threads.csv'.format(n_threads))
# plt.savefig('timings_{}_threads.png'.format(n_threads))
