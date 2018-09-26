import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import HawkesKernelExp, HawkesKernelExpLag, HawkesKernelSumExpLag
from tick.inference import HawkesEM

from tick.simulation import SimuHawkes


from tick.plot import plot_hawkes_kernels

run_time = 100000.0

associated_betas = np.array([1.0, 50.0])
associated_lags = np.array([0.5, 1.0])

associated_betas = np.array([5.0])
associated_lags = np.array([1.0])
U = len(associated_lags)


# kernel00 = HawkesKernelSumExpLag(np.array([0.2, 0.5]), associated_betas, associated_lags)
# kernel00 = HawkesKernelSumExpLag(np.array([0.1]), associated_betas, associated_lags)


baseline = np.array([0.4])
hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False, seed=2333)
hawkes.set_kernel(0, 0, kernel00)
hawkes.simulate()

em = HawkesEM(5, kernel_size=100, n_threads=8, verbose=False, tol=1e-5)
em.fit(hawkes.timestamps)

fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
plt.show()

# exit(0)




#------------------------------------------------------------------------------------------#
timestamps_list = []
global_n_list = []

n_nodes = 1
dim = n_nodes
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7])]

end_time = 2000.0
end_times = []

kernels = np.array([[
    kernel00]
])

for num_simu in range(200):
    seed = num_simu * 10086 + 666
    simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
    for i in range(n_nodes):
        simu_model.set_baseline(i, 0.4 + 0.1 * i)
        for j in range(n_nodes):
            simu_model.set_kernel(i, j, kernel00)
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
from tick.optim.model.hawkes_fixed_sumexpkern_lag_loglik_custom_list import ModelHawkesFixedSumExpKernLagCustomLogLikList
from tick.optim.solver import AGD
from tick.optim.prox import ProxZero, ProxL1


model_list = ModelHawkesFixedSumExpKernLagCustomLogLikList(associated_betas, associated_lags, MaxN_of_f, max_n_threads=8)
model_list.fit(timestamps_list, global_n_list, end_times=end_times)

prox = ProxZero()

solver = AGD(step=1e-3, linesearch=False, max_iter=5000, print_every=50)
solver.set_model(model_list).set_prox(prox)

x_real = np.array(
    [0.4,
                0.5,
                1, 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7
     ])

x0 = np.random.rand(model_list.n_coeffs)
solver.solve(x0)

print(model_list.loss(x_real))
print(model_list.loss(solver.solution))

#-------------------------------------------------------------------------------------------------------
coeff = solver.solution
Total_States = MaxN_of_f

for i in range(dim):
    fi0 = coeff[dim + dim * dim * U + i * Total_States]
    coeff[i] *= fi0
    for u in range(U):
        coeff[dim + dim * dim * u + i * dim: dim + dim * dim * u + (i + 1) * dim] *= fi0
    coeff[dim + dim * dim * U + i * Total_States: dim + dim * dim * U + (i + 1) * Total_States] /= fi0

print(coeff)
