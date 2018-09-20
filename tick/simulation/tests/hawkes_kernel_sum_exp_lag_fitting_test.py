import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import HawkesKernelExp, HawkesKernelExpLag, HawkesKernelSumExpLag, HawkesKernelSumExp
from tick.inference import HawkesEM

from tick.simulation import SimuHawkes


from tick.plot import plot_hawkes_kernels

run_time = 100000

baseline = np.array([0.2, 0.3])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2333)

beta = 1.0
betas = np.array([3.0, 5.0])

kernel1 = HawkesKernelSumExpLag(np.array([0.2, 0.2]), betas, np.array([0.5, 1.5]))
# kernel1 = HawkesKernelSumExp(np.array([0.2, 0.2]), betas)
kernel2 = HawkesKernelSumExpLag(np.array([0.2, 0.1]), betas, np.array([1.0, 1.0]))


hawkes.set_kernel(0, 0, kernel1)
hawkes.set_kernel(1, 0, kernel2)
hawkes.set_kernel(0, 1, HawkesKernelExp(0.3, 5))
hawkes.set_kernel(1, 1, HawkesKernelExp(0.3, 2))

hawkes.simulate()

em = HawkesEM(4, kernel_size=100, n_threads=8, verbose=False, tol=1e-5)
em.fit(hawkes.timestamps)

fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)

# for ax in fig.axes:
#     ax.set_ylim([0, 1])
plt.show()