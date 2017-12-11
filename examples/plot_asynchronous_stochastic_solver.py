"""
==============================
Asynchronous stochastic solver
==============================

This example illustrates the convergence speed of the asynchronous version of
SVRG solver. This solver called KroMagnon has been introduced in

Mania, H., Pan, X., Papailiopoulos, D., Recht, B., Ramchandran, K. and Jordan, M.I., 2015.
Perturbed iterate analysis for asynchronous stochastic optimization.
`arXiv preprint arXiv:1507.06970.`_.

.. _arXiv preprint arXiv:1507.06970.: https://arxiv.org/abs/1507.06970

To obtain good speedup in a relative short time example we have designed very
sparse and ill-conditonned problem.
"""


from scipy import sparse
import matplotlib.pyplot as plt
from tick.plot import plot_history
import numpy as np
from tick.optim.model import ModelLogReg
from tick.solver import SVRG
from tick.prox import ProxElasticNet
from tick.simulation import SimuLogReg, weights_sparse_gauss

seed = 1398
np.random.seed(seed)

n_samples = 40000
n_features = 20000
sparsity = 1e-4
penalty_strength = 1e-6

weights = weights_sparse_gauss(n_features, nnz=10)
intercept = 0.2
features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')

simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                       verbose=False, intercept=intercept)
features, labels = simulator.simulate()

model = ModelLogReg(fit_intercept=True)
model.fit(features, labels)
prox = ProxElasticNet(penalty_strength, ratio=0.5, range=(0, n_features))
svrg_step = 1. / model.get_lip_max()

test_n_threads = [1, 2, 4]

svrg_list = []
svrg_labels = []

for n_threads in test_n_threads:
    svrg = SVRG(step=svrg_step, seed=seed, max_iter=30, verbose=False,
                n_threads=n_threads)
    svrg.set_model(model).set_prox(prox)
    svrg.solve()

    svrg_list += [svrg]
    if n_threads == 1:
        svrg_labels += ['SVRG']
    else:
        svrg_labels += ['ASVRG {}'.format(n_threads)]

plot_history(svrg_list, x="time", dist_min=True, log_scale=True,
             labels=svrg_labels, show=False)
plt.ylim([3e-3, 0.3])
plt.ylabel('log distance to optimal objective', fontsize=14)
plt.tight_layout()
plt.show()

