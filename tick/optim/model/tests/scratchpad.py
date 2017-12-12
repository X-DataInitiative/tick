# License: BSD 3 clause

# from tick.optim.model import ModelHawkesCustom
#
# import numpy as np
# from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
# from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
# from tick.optim.model import ModelLogReg
# from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero
# from tick.plot import plot_history
#
# beta = 2.0
# MaxN_of_f = 5
# end_time = 4.5
#
#
# timestamps = [np.array([0.31, 0.93, 1.29, 2.32, 4.25]),
#               np.array([0.12, 1.19, 2.12, 2.41, 3.77, 4.21])]
#
# global_n = np.array([0,    1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1])
#
# coeffs = np.array([1., 3., 2., 3., 4., 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
#
# model = ModelHawkesCustom(beta, MaxN_of_f)
# model.fit(timestamps, global_n, end_time)
#
# print(model.loss(coeffs))
# print(model.grad(coeffs))


from tick.optim.model import ModelHawkesCustom

import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.model import ModelLogReg
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1
from tick.plot import plot_history

'''
pre_set parameters
'''
beta = 2.0
end_time = 10000

'''
generating a hawkes expnential process
'''


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
# global_n = np.array([0,    1,0,1,0,1,0,1,0,1,2,1])

timestamps, baseline, adjacency = get_train_data(n_nodes=2, betas=beta)

print(baseline)
print(adjacency)

# from tick.inference import HawkesExpKern
#
# decays = np.ones((2, 2)) * 2
# learner = HawkesExpKern(decays, penalty = 'l1', C = 100)
# learner.fit(timestamps)
#
# print(learner.baseline)
# print(learner.adjacency)

'''
calculate global_n and maxN_of_f
'''
global_n = np.random.randint(0, 9, size=1 + len(timestamps[0]) + len(timestamps[1]))
MaxN_of_f = 10
print(global_n)

'''
setting the inital point
'''
x0 = [baseline, adjacency.flatten(), np.ones(MaxN_of_f * 2) * 2]
x0 = np.array([i for row in x0 for i in row])  # flatten x0

'''
create a model_custom
'''
model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, global_n, end_time)

print('initial paras :', x0)
print('initial loss :', model.loss(x0))
print('initial grad :', model.grad(x0))

'''
optimization of parameters
'''
# prox = ProxL2Sq(0, positive=True)
prox = ProxL1(100, positive=True)

solvers = []
labels = []

steps = np.logspace(-4, -2.5, 2)
for step in steps:
    print(step)
    # solver = GD(step=step, linesearch = False, max_iter=2000, print_every=200)
    solver = AGD(linesearch=True)
    solver.set_model(model).set_prox(prox)

    try:
        solver.solve(x0)
    except RuntimeError as e:
        print(solver.solution)
        raise e

    solvers += [solver]
    labels += ['GD {:.3g}'.format(step)]

plot_history(solvers, labels=labels)
