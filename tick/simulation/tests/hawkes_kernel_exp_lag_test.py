import numpy as np
import matplotlib.pyplot as plt

from tick.simulation import HawkesKernelExp, HawkesKernelExpLag
from tick.inference import HawkesEM, HawkesConditionalLaw

from tick.simulation import SimuHawkes, HawkesKernelPowerLaw


from tick.plot import plot_hawkes_kernels

run_time = 30000

baseline = np.array([0.2, 0.3])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2333)

beta = 1.0

hawkes.set_kernel(0, 0, HawkesKernelExpLag(0.3, beta, 1.))
hawkes.set_kernel(0, 1, HawkesKernelExp(.5, 1.6))
hawkes.set_kernel(1, 1, HawkesKernelExp(0.3, beta))

hawkes.simulate()

em = HawkesEM(4, kernel_size=16, n_threads=8, verbose=False, tol=1e-3)
em.fit(hawkes.timestamps)

fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)

for ax in fig.axes:
    ax.set_ylim([0, 1])
plt.show()

# support = 4
# e = HawkesConditionalLaw(claw_method="log",
#                          delta_lag=0.1, min_lag=0.002, max_lag=100,
#                          quad_method="log",
#                          n_quad=50, min_support=0.002, max_support=support,
#                          n_threads=-1)
#
# e.incremental_fit(hawkes.timestamps)
# e.compute()
#
# fig = plot_hawkes_kernels(e, log_scale=True, hawkes=hawkes, show=False,
#                           min_support=0.002, support=100)
# for ax in fig.axes:
#     ax.legend(loc=3)
#     ax.set_ylim([1e-7, 1e2])
#
# plt.show()