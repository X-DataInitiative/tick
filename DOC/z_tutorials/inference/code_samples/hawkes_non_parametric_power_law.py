import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from tick.inference import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkes, HawkesKernelPowerLaw

rcParams['figure.figsize'] = (11, 7)

np.set_printoptions(precision=4)

multiplier = np.array([0.012, 0.008, 0.004, 0.0005])
cutoff = 0.0005
exponent = 1.3

hawkes = SimuHawkes(
    kernels=[[HawkesKernelPowerLaw(multiplier[0], cutoff, exponent, 2000),
              HawkesKernelPowerLaw(multiplier[1], cutoff, exponent, 2000)],
             [HawkesKernelPowerLaw(multiplier[2], cutoff, exponent, 2000),
              HawkesKernelPowerLaw(multiplier[3], cutoff, exponent, 2000)]],
    baseline=[0.05, 0.05], seed=382, verbose=False)
hawkes.end_time = 50000
hawkes.simulate()

e = HawkesConditionalLaw(claw_method="log",
                         delta_lag=0.1, min_lag=0.002, max_lag=100,
                         quad_method="log",
                         n_quad=50, min_support=0.002, max_support=2000)

e.incremental_fit(hawkes.timestamps)
e.compute()

# fig = plot_hawkes_kernels(e, log_scale=True, hawkes=hawkes, show=False,
#                           min_support=0.002)
# for ax in fig.axes:
#     ax.legend(loc=3)
#
# plt.show()
