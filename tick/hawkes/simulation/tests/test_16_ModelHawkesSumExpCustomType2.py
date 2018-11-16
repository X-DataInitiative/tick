# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN = 5
mu_i = [np.array([0.5, 0.7, 0.8, 0.6, 0.5]), np.array([0.5, 0.6, 0.8, 0.8, 0.6])]

end_time = 10000
betas = np.array([1.0, 3, 15, 100])

U = len(betas)
kernels = np.array([
            [HawkesKernelSumExp(np.array([0.2, 0.15, 0.1, 0.1]), betas), HawkesKernelSumExp(np.array([0.3, 0, 0.1, 0.1]), betas)],
            [HawkesKernelSumExp(np.array([0., 0.2, 0.2, 0.0]), betas), HawkesKernelSumExp(np.array([0., 0.4, 0, 0.1]), betas)]
        ])

simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom='Type2', seed=seed, MaxN_of_f = MaxN, f_i=mu_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.0)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()

# plot_point_process(simu_model)
print("nombre de point :", len(simu_model.timestamps[0]), len(simu_model.timestamps[1]))
##################################################################################################################





##################################################################################################################
from tick.optim.model import ModelHawkesSumExpCustomType2
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1

timestamps = simu_model.timestamps
timestamps.append(np.array([]))
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)

model = ModelHawkesSumExpCustomType2(betas, MaxN, n_threads=4)
model.fit(timestamps, global_n, end_time)
#############################################################################
prox = ProxZero()

# solver = AGD(step=5e-2, linesearch=False, max_iter= 350)
solver = AGD(step=0.1, linesearch=False, max_iter=5000, print_every=50)
solver.set_model(model).set_prox(prox)

x_real = np.array(
    [0.5, 0.7, 0.8, 0.6, 0.5,  0.5, 0.6, 0.8, 0.8, 0.6,
     0.2, 0.3, 0, 0,  0.15, 0, 0.2, 0.4,  0.1, 0.1, 0.2, 0, 0.1, 0.1, 0, 0.1])
x0 = np.array(
    [0.2, 0.3, 0.4, 0.2, 0.6,  0.2, 0.1, 0.2, 0.9, 0.3,
     0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
solver.solve(x0)
# solver.solve(x_real)

print(model.loss(x_real))
print(model.loss(solver.solution))
print(solver.solution)
