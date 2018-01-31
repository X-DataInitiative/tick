# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

end_times = np.array([10.0, 10.0])
timestamps_list = []
global_n_list = []

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7]), np.array([1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])]

end_time = 10
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
simu_model.simulate()

timestamps = simu_model.timestamps
timestamps.append(np.array([3]))

timestamps_list.append(timestamps)
print(timestamps_list)

global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)
global_n_list.append(global_n)
##################################################################################################################
seed = 8006

simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.4 + 0.1 * i)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
simu_model.simulate()

timestamps = simu_model.timestamps
timestamps.append(np.array([4]))

timestamps_list.append(timestamps)
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)
global_n_list.append(global_n)


##################################################################################################################
from tick.optim.model.hawkes_fixed_sumexpkern_loglik_custom_list import ModelHawkesFixedSumExpKernCustomLogLikList

model_list = ModelHawkesFixedSumExpKernCustomLogLikList(betas, MaxN_of_f)
model_list.fit(timestamps_list, np.array(global_n_list), end_times=end_times)
