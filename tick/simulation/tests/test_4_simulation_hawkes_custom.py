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
f_i = [np.array([1.0, 1.0,1.0,1.0, 1.0]) * 2, np.array([1.0, 1.0,1.0,1.0, 1.0]) * 2]

kernels = np.array([
            [HawkesKernelExp(0.6, 3), HawkesKernelExp(0.6, 3)],
            [HawkesKernelExp(0.2, 3), HawkesKernelExp(0.3, 3)]
        ])

kernels_2 = np.array([
                [HawkesKernelExp(0.3, 3), HawkesKernelExp(0.3, 3)],
             [HawkesKernelExp(0.1, 3), HawkesKernelExp(0.15, 3)]
        ])

kernels_3 = np.array([
            [HawkesKernelExp(0.6, 3), HawkesKernelExp(0.6, 3)],
            [HawkesKernelExp(0.2, 3), HawkesKernelExp(0.3, 3)]
        ])


simu_model = SimuHawkes(kernels=kernels_2, end_time=10, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.15)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels_2[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()
plot_point_process(simu_model)


simu_model2 = SimuHawkes(kernels=kernels, end_time=10, seed=seed)
for i in range(n_nodes):
    simu_model2.set_baseline(i, 0.3)
    for j in range(n_nodes):
        simu_model2.set_kernel(i, j, kernels[i, j])

simu_model2.track_intensity(0.1)
# print(simu_model.simulate)
simu_model2.simulate()
plot_point_process(simu_model2)

print('#'*40)
print('Hawkes Custom:\n', simu_model.kernels, simu_model.baseline, '\n', np.array(simu_model.timestamps))
print(simu_model._pp.get_global_n())
print('Hawkes Pure:\n', simu_model2.kernels, simu_model2.baseline, '\n', simu_model2.timestamps)
