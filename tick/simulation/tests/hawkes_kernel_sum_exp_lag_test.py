import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import HawkesKernelExp, HawkesKernelExpLag, HawkesKernelSumExpLag, HawkesKernelSumExp
from tick.inference import HawkesEM, HawkesConditionalLaw

from tick.simulation import SimuHawkes


from tick.plot import plot_hawkes_kernels

run_time = 100000

baseline = np.array([0.4, 0.4])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2333)

beta = 1.0
betas = np.array([5.0, 5.0])

kernel00 = HawkesKernelSumExpLag(np.array([0.1]), np.array([5.0]), np.array([1.0]))
kernel1 = HawkesKernelSumExp(np.array([0.2, 0.2]), betas)
kernel10 = HawkesKernelSumExpLag(np.array([0.2, 0.1]), betas, np.array([1.0, 1.0]))
kernel01 = HawkesKernelExp(0.3, 5)
kernel11 = HawkesKernelExp(0.3, 2)



hawkes.set_kernel(1, 1, kernel10)
hawkes.set_kernel(0, 0, kernel10)
hawkes.set_kernel(1, 0, kernel10)
hawkes.set_kernel(0, 1, kernel10)

print(hawkes.get_baseline_values(0,0))
print(hawkes.get_baseline_values(1,0))
print(hawkes.kernels)

hawkes.simulate()

em = HawkesEM(5, kernel_size=100, n_threads=8, verbose=False, tol=1e-5)
em.fit(hawkes.timestamps)

print(em.baseline)
fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
plt.show()

#################################################################################################

from tick.plot import plot_basis_kernels, plot_hawkes_kernels
from tick.inference import HawkesBasisKernels

ticks = hawkes.timestamps

# And then perform estimation with two basis kernels
kernel_support = 5
n_basis = 2

em = HawkesBasisKernels(kernel_support, n_basis=n_basis,
                        kernel_size=50,
                        n_threads=4, max_iter=100,
                        verbose=False, ode_tol=1e-5)
em.fit(ticks)

fig = plot_hawkes_kernels(em, hawkes=hawkes, support=10, show=False)

# fig = plot_basis_kernels(em, basis_kernels=[g2, g1], show=False)

plt.show()