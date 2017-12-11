# License: BSD 3 clause

from tick.optim.model import ModelHawkesCustom

import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.model import ModelLogReg
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero
from tick.plot import plot_history

beta = 2.0
MaxN_of_f = 10
end_time = 10


def get_train_data(n_nodes, betas):
    np.random.seed(130947)
    baseline = np.random.rand(n_nodes)
    adjacency = np.random.rand(n_nodes, n_nodes)
    if isinstance(betas, (int, float)):
        betas = np.ones((n_nodes, n_nodes)) * betas

    sim = SimuHawkesExpKernels(adjacency=adjacency, decays=betas,
                               baseline=baseline, verbose=False,
                               seed=13487, end_time=end_time)
    sim.adjust_spectral_radius(0.5)
    adjacency = sim.adjacency
    sim.simulate()

    return sim.timestamps, baseline, adjacency

# timestamps = [np.array([0.31, 0.93, 1.29, 2.32, 4.25, 4.35, 4.78, 5.5, 6.83, 6.99]),
#               np.array([0.12, 1.19, 2.12, 2.41, 3.77, 4.21, 4.96, 5.11, 6.7, 7.26])]

global_n = np.array([0,    1,0,1,0,1,0,1,0,1,2,1])

timestamps, baseline, adjacency = get_train_data(n_nodes=2, betas=beta)

# coeffs = np.array([1., 3., 2., 3., 4., 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])

model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, end_time)
prox = ProxL2Sq(1, positive=True)

solver_params = {'max_iter': 100, 'tol': 1e-6, 'verbose': False}
# x0 = np.array([1., 3., 2., 3., 4., 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
x0 = 1 * np.ones(model.n_coeffs)

# sgd = SGD(**solver_params).set_model(model).set_prox(prox)
# sgd.solve(x0, step=500.)


solvers = []
labels = []

steps = np.logspace(-4, -2.5, 2)

for step in steps:
    print(step)
    solver = GD(step=step, linesearch=False, max_iter=2000, print_every=200)
    solver.set_model(model).set_prox(prox)

    try:
        solver.solve(x0)
    except RuntimeError as e:
        print(solver.solution)
        raise e

    solvers += [solver]
    labels += ['AGD {:.3g}'.format(step)]

plot_history(solvers, labels=labels)
