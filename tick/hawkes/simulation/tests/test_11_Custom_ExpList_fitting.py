# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

timestamps_list = []
global_n_list = []

n_nodes = 3
dim = n_nodes
MaxN_of_f = 5
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5]), np.array([1., 0.6, 0.8, 0.8, 0.6]), np.array([1., 0.6, 0.9, 0.2, 0.7])]

end_time = 20.0
end_times = []

beta = 3
kernels = np.array([
            [HawkesKernelExp(0.3, beta), HawkesKernelExp(0.1, beta), HawkesKernelExp(0.4, beta)],
            [HawkesKernelExp(0.2, beta), HawkesKernelExp(0.3, beta), HawkesKernelExp(0.5, beta)],
            [HawkesKernelExp(0.3, beta), HawkesKernelExp(0.4, beta), HawkesKernelExp(0.3, beta)]
])

for num_simu in range(10000):
    seed = num_simu * 1086 + 3007
    simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
    for i in range(n_nodes):
        simu_model.set_baseline(i, 0.2 + 0.1 * i)
        for j in range(n_nodes):
            simu_model.set_kernel(i, j, kernels[i, j])
    simu_model.track_intensity(0.1)
    simu_model.simulate()

    timestamps = simu_model.timestamps
    timestamps.append(np.array([]))
    timestamps_list.append(timestamps)

        # print(len(timestamps[0]) + len(timestamps[1])+len(timestamps[2]))
        # exit(0)

    global_n = np.array(simu_model._pp.get_global_n())
    global_n = np.insert(global_n, 0, 0).astype(int)
    global_n_list.append(global_n)

    end_times.append(end_time)

end_times = np.array(end_times)
##################################################################################################################
from tick.optim.model.hawkes_fixed_expkern_loglik_custom_list import ModelHawkesFixedExpKernCustomLogLikList

model_list = ModelHawkesFixedExpKernCustomLogLikList(beta, MaxN_of_f, n_threads = 8)
# model_list = ModelHawkesFixedExpKernCustomLogLikList(beta, MaxN_of_f)
model_list.fit(timestamps_list, global_n_list, end_times=end_times)



##################################################################################################################
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1
prox = ProxL1(0.0, positive=True)
prox = ProxZero()

solver = AGD(step=1e-3, linesearch=False, max_iter=10000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.2, 0.3, 0.4,  0.3, 0.1, 0.4, 0.2, 0.3, 0.5, 0.3, 0.4, 0.3,   0.7, 0.8, 0.6, 0.5, 0.6, 0.8, 0.8, 0.6, 0.6, 0.9, 0.2, 0.7])
x0 = np.array(
    [0.6, 0.6, 0.8,  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6,   0.5, 0.5, 0.9, 0.9, 0.6, 0.7, 0.8, 0.5, 0.7, 0.7, 0.5, 0.5])
solver.solve(x0)

print(model_list.loss(x_real))
print(model_list.loss(solver.solution))
print(solver.solution/x_real)

###########################################################################################
# manuel grad check
# delta = 1e-6
# x1 = x0.copy()
# grad2 = []
# for i in range(len(x0)):
#     x1[i] += delta
#     grad2.append((model_list.loss(x1) - model_list.loss(x0)) / delta)
#     x1[i] -= delta
# print(model_list.grad(x0) / grad2)


# from tick.optim.model import ModelHawkesCustom
# tmp1 = 0
# tmp2 = 0
# tmp3 = 0
# for i in range(20):
#     timestamps = timestamps_list[i]
#     global_n = global_n_list[i]
#     model = ModelHawkesCustom(beta, MaxN_of_f)
#     model.fit(timestamps, global_n, end_times[i])
#     tmp1 += model.loss(x0) * (len(timestamps_list[i][0]) + len(timestamps_list[i][1]) + len(timestamps_list[i][2]))
#     tmp2 += (len(timestamps_list[i][0]) + len(timestamps_list[i][1]) + len(timestamps_list[i][2]))
#     tmp3 += model.grad(x0) * (len(timestamps_list[i][0]) + len(timestamps_list[i][1]) +len(timestamps_list[i][2]))
#
# print("Loss calculated using list:", model_list.loss(x0))
# print("Loss calculated accurate  :", tmp1 / tmp2)
# print("grad calculated using list:", model_list.grad(x0))
# print("grad calculated accurate  :", tmp3 / tmp2)
# print(model_list.grad(x0) / (tmp3 / tmp2))
