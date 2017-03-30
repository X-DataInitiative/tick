import numpy as np
import matplotlib.pyplot as plt

from tick.inference import HawkesEM
from tick.simulation import SimuHawkes, HawkesKernelTimeFunc
from tick.base import TimeFunction
from tick.plot import plot_hawkes_kernels

run_time = 10000

t_values1 = np.array([0, 1, 1.5, 2., 3.5], dtype=float)
y_values1 = np.array([0, 0.2, 0, 0.1, 0.], dtype=float)
tf1 = TimeFunction([t_values1, y_values1],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel1 = HawkesKernelTimeFunc(tf1)

t_values2 = np.array([0, 2, 2.5], dtype=float)
y_values2 = np.array([0, 0.6, 0], dtype=float)
tf2 = TimeFunction([t_values2, y_values2],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel2 = HawkesKernelTimeFunc(tf2)

baseline = np.array([0.1, 0.3])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2334)

hawkes.set_kernel(0, 0, kernel1)
hawkes.set_kernel(0, 1, kernel1)
hawkes.set_kernel(1, 1, kernel2)

hawkes.simulate()

em = HawkesEM(4, kernel_size=8, n_threads=8, verbose=True, tol=1e-3)
em.fit(hawkes.timestamps)

fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)

for ax in fig.axes:
    ax.set_ylim([0, 1])
plt.show()
