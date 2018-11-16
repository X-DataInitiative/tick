# License: BSD 3 clause

from tickmodel import ModelHawkesCustom

import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
from ticksolver import GD, AGD, SGD, SVRG, SDCA
from tickprox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1
from tick.plot import plot_history

'''
pre_set parameters
'''
beta = 2.0
end_time = 10000
dim = 2

'''
generating a hawkes expnential process
'''


def get_train_data(n_nodes, betas):
    np.random.seed(330676)
    baseline = np.random.rand(n_nodes)
    adjacency = np.random.rand(n_nodes, n_nodes)
    if isinstance(betas, (int, float)):
        betas = np.ones((n_nodes, n_nodes)) * betas

    sim = SimuHawkesExpKernels(adjacency=adjacency, decays=betas,
                               baseline=baseline, verbose=False,
                               seed=13487, end_time=end_time)
    sim.adjust_spectral_radius(0.8)
    adjacency = sim.adjacency
    sim.simulate()

    return sim.timestamps, baseline, adjacency


timestamps, baseline, adjacency = get_train_data(n_nodes=dim, betas=beta)

print('data size =', len(timestamps[0]), ',', len(timestamps[1]))

print(baseline)
print(adjacency)

from tick.inference import HawkesExpKern

decays = np.ones((dim, dim)) * beta
learner = HawkesExpKern(decays, penalty='l1', C=100)
learner.fit(timestamps)

print('#' * 40)
print(learner.baseline)
print(learner.adjacency)

print('#' * 40)
'''
calculate global_n and maxN_of_f
'''
MaxN_of_f = 10
global_n = np.random.randint(0, MaxN_of_f - 1, size=1 + len(timestamps[0]) + len(timestamps[1]))
'''
create a model_custom
'''
model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, global_n, end_time)
#############################################################################
prox = ProxL1(0.01, positive=True)

# solver = AGD(step=5e-2, linesearch=False, max_iter= 350)
solver = AGD(step=1e-2, linesearch=False, max_iter=2000)
solver.set_model(model).set_prox(prox)

# x0_3 = np.array([1,2,3,1,2,3.0,   1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
# x0_3 = np.array([1.0,1,1,1,1,1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# x0_3 = np.array([0.55, 0.5, 0.06, 0.09, 0.8, 0.9,    1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
# x0_3 = np.array([0.55, 0.5, 0.06, 0.09, 0.8, 0.9,   0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1])
# x0_3 = np.array([0.55, 0.5, 0.06, 0.09, 0.8, 0.9,   2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4])
x0_3 = np.array([2, 3, 2, 3, 2, 3.0, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4])
solver.solve(x0_3)

# normalisation
solution_adj = solver.solution
for i in range(dim):
    solution_adj[i] *= solver.solution[dim + dim * dim + MaxN_of_f * i]
    solution_adj[(dim + dim * i): (dim + dim * (i + 1))] *= solver.solution[dim + dim * dim + MaxN_of_f * i]
    solution_adj[(dim + dim * dim + MaxN_of_f * i): (dim + dim * dim + MaxN_of_f * (i + 1))] /= solver.solution[
        dim + dim * dim + MaxN_of_f * i]
print(solution_adj)
#############################################################################
'''
optimization of parameters
'''
# # prox = ProxL2Sq(0, positive=True)
# prox = ProxL1(100, positive=True)
#
# solvers = []
# labels = []
#
# x0_3 = np.array([ 0.55,  0.5,  0.06,  0.09,  0.8, 0.9, 1,1,1,1,1,1,1,1,1,1,   2,2,2,2,2,2,2,2,2,2])
#
# steps = np.logspace(-4, -2.5, 2)
# for step in steps:
#     print(step)
#     # solver = GD(step=step, linesearch = False, max_iter=2000, print_every=200)
#     solver = GD(linesearch=True)
#     solver.set_model(model).set_prox(prox)
#
#     try:
#         solver.solve(x0_3)
#     except RuntimeError as e:
#         print(solver.solution)
#         raise e
#
#     solvers += [solver]
#     labels += ['GD {:.3g}'.format(step)]
#
# plot_history(solvers, labels=labels)
