# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7]), np.array([1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])]

end_time = 10000
betas = np.array([0.1, 1, 3, 10])

U = len(betas)
kernels = np.array([
            [HawkesKernelSumExp(np.array([0.2, 0.15, 0.1, 0.1]), betas), HawkesKernelSumExp(np.array([0.3, 0, 0.1, 0.1]), betas)],
            [HawkesKernelSumExp(np.array([0., 0.2, 0.2, 0.0]), betas), HawkesKernelSumExp(np.array([0., 0.4, 0, 0.1]), betas)]
        ])

simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.4 + 0.1 * i)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()

##################################################################################################################





##################################################################################################################
from tick.optim.model import ModelHawkesSumExpCustom
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1

timestamps = simu_model.timestamps
timestamps.append(np.array([]))
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)


##################################################################################################################
from tick.optim.model.hawkes_fixed_sumexpkern_lag_loglik_custom import ModelHawkesSumExpCustomLag
from tick.optim.solver import AGD
from tick.optim.prox import ProxZero, ProxL1


associated_betas = betas
associated_lags = np.zeros(len(betas))
model_list = ModelHawkesSumExpCustomLag(associated_betas, associated_lags, MaxN_of_f, max_n_threads=8)
model_list.fit(timestamps, global_n, end_times=end_time)

prox = ProxZero()

solver = AGD(step=1e-3, linesearch=False, max_iter=10000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.4, 0.5,   0.2, 0.3, 0, 0,  0.15, 0, 0.2, 0.4,  0.1, 0.1, 0.2, 0, 0.1, 0.1, 0, 0.1,
     1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7,  1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])

x0 = np.random.rand(model_list.n_coeffs)
solver.solve(x0)

print(model_list.loss(x_real))
print(model_list.loss(solver.solution))
# print(solver.solution/x_real)

np.save("sumexplagfall.npy", solver.solution)
