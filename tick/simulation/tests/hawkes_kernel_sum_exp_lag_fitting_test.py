import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import HawkesKernelExp, HawkesKernelExpLag, HawkesKernelSumExpLag
from tick.inference import HawkesEM

from tick.simulation import SimuHawkes


from tick.plot import plot_hawkes_kernels

# run_time = 100000
#
# baseline = np.array([0.2, 0.3])
#
# hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
#                     seed=2333)

beta = 2.0
betas = np.array([3.0, 5.0])

kernel00 = HawkesKernelSumExpLag(np.array([0.2, 0.5]), betas, np.array([0.5, 1.5]))
kernel01 = HawkesKernelExpLag(0.3, beta, 1.)
kernel10 = HawkesKernelExp(0.3, 5)
kernel11 = HawkesKernelExp(0.3, beta)

# hawkes.set_kernel(0, 0, kernel00)
# hawkes.set_kernel(1, 0, kernel10)
# hawkes.set_kernel(0, 1, kernel01)
# hawkes.set_kernel(1, 1, kernel11)

# hawkes.simulate()

# em = HawkesEM(4, kernel_size=100, n_threads=8, verbose=False, tol=1e-5)
# em.fit(hawkes.timestamps)
#
# fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
#
# for ax in fig.axes:
#     ax.set_ylim([0, 1])
# plt.show()
#
# exit(0)




#------------------------------------------------------------------------------------------#
n_nodes = 2
dim = n_nodes
MaxN_of_f = 5
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5]), np.array([1., 0.6, 0.8, 0.8, 0.6])]

end_time = 10000.0

kernels = np.array([
    [kernel00, kernel01],
    [kernel10, kernel11]
])

seed = 66
simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.2 + 0.1 * i)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
simu_model.simulate()

timestamps = simu_model.timestamps
timestamps.append(np.array([]))

global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)

#7637 7638

##################################################################################################################
from tick.optim.model.hawkes_fixed_sumexpkern_lag_loglik_custom import ModelHawkesSumExpCustomLag
from tick.optim.solver import AGD
from tick.optim.prox import ProxZero, ProxL1


associated_betas = np.array([2, 2, 3.0, 5, 5])
associated_lags = np.array([2, 2, 3.0, 5, 5])
model_list = ModelHawkesSumExpCustomLag(associated_betas, associated_lags, MaxN_of_f, max_n_threads=8)
model_list.fit(timestamps, global_n, end_times=end_time)

prox = ProxZero()

solver = AGD(step=1e-3, linesearch=False, max_iter=15000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.2, 0.3,  0.0, 0.0, 0.2, 0.0, 0.5,            0.0, 0.3, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.3, 0.0,            0.3, 0.0, 0.0, 0.0, 0.0,

                1., 0.7, 0.8, 0.6, 0.5,				1., 0.6, 0.8, 0.8, 0.6])

x_real = np.array(
    [0.2, 0.3,  0.0, 0.0, 0.0, 0.3,
                0.0, 0.3, 0.0, 0.0,
                0.2, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.3, 0.0,
                0.5, 0.0, 0.0, 0.0,
                1., 0.7, 0.8, 0.6, 0.5,				1., 0.6, 0.8, 0.8, 0.6])

x0 = np.random.rand(model_list.n_coeffs)
solver.solve(x0)

print(model_list.loss(x_real))
print(model_list.loss(solver.solution))
# print(solver.solution/x_real)

np.save("sumexplag.npy", solver.solution)
