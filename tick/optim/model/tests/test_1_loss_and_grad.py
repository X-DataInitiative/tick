# License: BSD 3 clause

from tick.optim.model import ModelHawkesCustom

import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
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
    np.random.seed(256707)
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


timestamps, baseline, adjacency = get_train_data(n_nodes=2, betas=beta)

print('data size =', len(timestamps[0]), ',', len(timestamps[1]))

print(baseline)
print(adjacency)

from tick.inference import HawkesExpKern

decays = np.ones((2, 2)) * beta
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
setting the inital point
'''
x0 = [baseline, adjacency.flatten(), np.ones(MaxN_of_f * 2)]
x0 = np.array([i for row in x0 for i in row])  # flatten x0

'''
create a model_custom
'''
model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, global_n, end_time)

print('custom paras :', x0)
print('custom loss :', model.loss(x0))
print('custom grad :', model.grad(x0))

from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesExpKern
from tick.optim.model import ModelHawkesFixedExpKernLogLik

'''
compare with model pure hawkes
'''
x0_2 = np.concatenate((baseline, adjacency.flatten()))
model2 = ModelHawkesFixedExpKernLogLik(beta)
model2.fit(timestamps, end_time)

print('hawkes paras :', x0_2)
print('hawkes loss :', model2.loss(x0_2))
print('hawkes grad :', model2.grad(x0_2))
