# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

from tick.plot import plot_point_process

n_nodes = 2
seed = 3007
MaxN_of_f = 5
f_i = [np.array([1.0, 7, 7.7, 6, 3]), np.array([1.0, 0.5, 2, 1, 2])]

kernels = np.array([
            [HawkesKernel0(), HawkesKernelExp(0.6, 3)],
            [HawkesKernelExp(0.2, 3), HawkesKernelExp(0.3, 3)]
        ])




simu_model = SimuHawkes(n_nodes = n_nodes, kernels=kernels, end_time=10, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.3)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()

print(simu_model.timestamps)
print(simu_model.baseline)

plot_point_process(simu_model)


# simu_model2 = SimuHawkes(kernels=kernels, end_time=10, seed=seed)
# for i in range(n_nodes):
#     simu_model2.set_baseline(i, 0.3)
#     for j in range(n_nodes):
#         simu_model2.set_kernel(i, j, kernels[i, j])
#
# simu_model2.track_intensity(0.1)
# # print(simu_model.simulate)
# simu_model2.simulate()
#
# print(simu_model2.timestamps)
# print(simu_model2.baseline)
#
# plot_point_process(simu_model2)
