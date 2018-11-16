# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN_of_f = 5
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5]), np.array([1., 0.6, 0.8, 0.8, 0.6])]

beta = 3.0
end_time = 100000

kernels = np.array([
            [HawkesKernelExp(0.7, beta), HawkesKernelExp(0.6, beta)],
            [HawkesKernelExp(0.2, beta), HawkesKernelExp(0.3, beta)]
        ])

simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.7)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()

##################################################################################################################
from tickmodel import ModelHawkesCustom

timestamps = simu_model.timestamps
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)

model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, global_n, end_time)

x0 = np.array(
    [0.6, 0.8, 0.5, 0.6, 0.7, 0.8,   1., 0.5, 0.5, 0.9, 0.9, 1., 0.6, 0.7, 0.8, 0.5])
print(model.loss(x0))
print(model.grad(x0))
##################################################################################################################
from tickmodel import ModelHawkesSumExpCustom

betas = np.array([beta, beta])

model2 = ModelHawkesSumExpCustom(betas, MaxN_of_f)
model2.fit(timestamps, global_n, end_time)
x0 = np.array(
    [0.6, 0.8,    0.25, 0.3, 0.35, 0.4, 0.25, 0.3, 0.35, 0.4,     1., 0.5, 0.5, 0.9, 0.9, 1., 0.6, 0.7, 0.8, 0.5])
x0 = np.array(
    [0.6, 0.8,    0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8,     1., 0.5, 0.5, 0.9, 0.9, 1., 0.6, 0.7, 0.8, 0.5])
print(model2.loss(x0))
print(model2.grad(x0))

#manuel grad
delta = 1e-8
x1 = x0.copy()
grad2 = []
for i in range(len(x0)):
    x1[i] += 1e-8
    grad2.append((model2.loss(x1) - model2.loss(x0)) / delta)
    x1[i] -= 1e-8
print(model2.grad(x0) / grad2)
