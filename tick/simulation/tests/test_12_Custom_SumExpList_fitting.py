# License: BSD 3 clause

import numpy as np
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

timestamps_list = []
global_n_list = []

n_nodes = 2
dim = n_nodes
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7]), np.array([1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])]

end_time = 2000.0
end_times = []

betas = np.array([100.0, 500, 5000])
U = len(betas)
kernels = np.array([
    [HawkesKernelSumExp(np.array([0.15, 0.1, 0.1]), betas), HawkesKernelSumExp(np.array([0, 0.1, 0.1]), betas)],
    [HawkesKernelSumExp(np.array([0.2, 0.2, 0.0]), betas), HawkesKernelSumExp(np.array([0.4, 0, 0.1]), betas)]
])

for num_simu in range(100):
    seed = num_simu * 10086 + 6666
    simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
    for i in range(n_nodes):
        simu_model.set_baseline(i, 0.4 + 0.1 * i)
        for j in range(n_nodes):
            simu_model.set_kernel(i, j, kernels[i, j])
    simu_model.track_intensity(0.1)
    simu_model.simulate()

    timestamps = simu_model.timestamps
    timestamps.append(np.array([]))
    timestamps_list.append(timestamps)

    global_n = np.array(simu_model._pp.get_global_n())
    global_n = np.insert(global_n, 0, 0).astype(int)
    global_n_list.append(global_n)

    end_times.append(end_time)

end_times = np.array(end_times)
##################################################################################################################
from tick.optim.model.hawkes_fixed_sumexpkern_loglik_custom_list import ModelHawkesFixedSumExpKernCustomLogLikList

model_list = ModelHawkesFixedSumExpKernCustomLogLikList(betas, MaxN_of_f, n_threads=8)
model_list.fit(timestamps_list, global_n_list, end_times=end_times)
x0 = np.array(
    [0.6, 0.8,
     0.2,0.2,0.2,0.2,  0.4,0.4,0.4,0.4, 0.5,0.5,0.5,0.5,
     0.5, 0.5, 0.5, 0.9, 0.9, 0.3, 0.6, 0.7, 0.8, 0.5,
     0.6, 0.5, 0.5, 0.9, 0.9, 0.3, 0.6, 0.7, 0.8, 0.5])

##################################################################################################################
from tick.optim.solver import AGD
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1
prox = ProxL1(0.0, positive=True)
prox = ProxZero()

solver = AGD(step=1e-4, linesearch=False, max_iter=20000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.4, 0.5,   0.15, 0, 0.2, 0.4,  0.1, 0.1, 0.2, 0, 0.1, 0.1, 0, 0.1,
     1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7,  1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])
# solver.solve(x0)

coeff = np.load("agd_10000.npy")
coeff_unadj = coeff.copy()

for i in range(dim):
    fi0 = coeff[dim + U * dim * dim + i * MaxN_of_f]
    coeff[i] *= fi0
    coeff[dim + U * dim * dim + i * MaxN_of_f: dim + U * dim * dim + (i + 1) * MaxN_of_f] /= fi0
    for j in range(dim):
        for u in range(U):
            coeff[dim + u * dim * dim + i * dim + j] *= fi0

print(model_list.loss(x_real))
print(model_list.loss(coeff_unadj))
print(model_list.loss(coeff))
print(x_real/coeff)
