# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN = 10
mu_i = [np.array([0.5, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6, 0.8, 0.1]), np.array([0.5, 0.6, 0.8, 0.8, 0.6, 0.7, 0.8, 0.6, 0.5, 0.4])]

end_time = 50.0
betas = np.array([5.0, 100])

U = len(betas)
kernels = np.array([
            [HawkesKernelSumExp(np.array([0.2, 0.1]), betas), HawkesKernelSumExp(np.array([0.3, 0.1]), betas)],
            [HawkesKernelSumExp(np.array([0.2, 0.05]), betas), HawkesKernelSumExp(np.array([0., 0.1]), betas)]
        ])
timestamps_list = []
global_n_list = []
end_times = []

for num_simu in range(100000):
    seed = num_simu * 10086 + 3007
    simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom='Type2', seed=seed, MaxN_of_f=MaxN, f_i=mu_i)

    for i in range(n_nodes):
        simu_model.set_baseline(i, 0.0)
        for j in range(n_nodes):
            simu_model.set_kernel(i, j, kernels[i, j])
    simu_model.track_intensity(0.1)
    simu_model.simulate()

    timestamps = simu_model.timestamps

    timestamps.append(np.array([]))
    timestamps_list.append(timestamps)

    global_n = np.array(simu_model._pp.get_global_n())
    global_n = np.insert(global_n, 0, num_simu % 5).astype(int)
    global_n_list.append(global_n)

    end_times.append(end_time)

end_times = np.array(end_times)
##################################################################################################################





##################################################################################################################
from tick.optim.solver import AGD
from tick.optim.prox import ProxZero, ProxL1
prox = ProxL1(0.0, positive=True)
prox = ProxZero()

from tick.optim.model.hawkes_fixed_sumexpkern_loglik_custom2_list import ModelHawkesFixedSumExpKernCustomType2LogLikList
model_list = ModelHawkesFixedSumExpKernCustomType2LogLikList(betas, MaxN, n_threads=8)
model_list.fit(timestamps_list, global_n_list, end_times=end_times)

solver = AGD(step=1e-2, linesearch=False, max_iter=5000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.5, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6, 0.8, 0.1,     0.5, 0.6, 0.8, 0.8, 0.6, 0.7, 0.8, 0.6, 0.5, 0.4,
     0.2, 0.3, 0.2, 0, 0.1, 0.1, 0.05, 0.1])
x0 = np.array(
    [0.5, 0.6, 0.2, 0.3, 0.8, 0.5, 0.7, 0.8, 0.6, 0.5,     0.5, 0.6, 0.2, 0.3, 0.8, 0.5, 0.7, 0.8, 0.6, 0.5,
     0.7, 0.5, 0.2, 0.3, 0.75, 0.1, 0.1, 0.2])

solver.solve(x0)

print(model_list.loss(x_real))
print(model_list.loss(solver.solution))
print(solver.solution)
