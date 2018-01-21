# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

end_times = [100000, 100000]
timestamps_list = []
global_n_list = []

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7]), np.array([1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])]

end_time = 100000
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

timestamps_list.aappend(simu_model.timestamps)
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

timestamps_list.append(simu_model.timestamps)
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)
global_n_list.append(global_n)



##################################################################################################################
from tick.optim.model.hawkes_fixed_sumexpkern_loglik_custom_list import ModelHawkesFixedSumExpKernCustomLogLikList

model_list = ModelHawkesFixedSumExpKernCustomLogLikList(betas, MaxN_of_f)
model_list.fit(timestamps_list, end_times=end_times)
def fit(self, events, global_n, end_times=None):


intensities_list = [
    hawkes_sumexp_kernel_intensities(
        baseline, decays, adjacency, timestamps)
    for timestamps in timestamps_list
]

integral_approx = sum([hawkes_log_likelihood(intensities,
                                             timestamps, end_time)
                       for (intensities, timestamps, end_time) in zip(
        intensities_list, timestamps_list,
        model_list.end_times
    )])

integral_approx /= model_list.n_jumps
assertAlmostEqual(integral_approx,
                       - model_list.loss(coeffs),
                       places=2)
