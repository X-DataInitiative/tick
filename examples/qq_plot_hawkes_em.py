import numpy as np
import matplotlib.pyplot as plt

from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp,
                         HawkesEM)
from tick.base import TimeFunction
from tick.plot import qq_plots

run_time = 30000

t_values1 = np.array([0, 1, 1.5, 2., 3.5], dtype=float)
y_values1 = np.array([0, 0.2, 0, 0.1, 0.], dtype=float)
tf1 = TimeFunction([t_values1, y_values1],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel1 = HawkesKernelTimeFunc(tf1)

t_values2 = np.linspace(0, 4, 20)
y_values2 = np.maximum(0., np.sin(t_values2) / 4)
tf2 = TimeFunction([t_values2, y_values2])
kernel2 = HawkesKernelTimeFunc(tf2)

baseline = np.array([0.1, 0.3])

hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False,
                    seed=2334)

hawkes.set_kernel(0, 0, kernel1)
hawkes.set_kernel(0, 1, HawkesKernelExp(.5, .7))
hawkes.set_kernel(1, 1, kernel2)

hawkes.simulate()

em = HawkesEM(4, kernel_size=16, n_threads=8, verbose=False, tol=1e-3)
em.fit(hawkes.timestamps)

hawkes.store_compensator_values()
residuals_list = em.time_changed_interarrival_times()

fig, axs = plt.subplots(2, 2)

_ = qq_plots(
    point_process=hawkes,
    ax=[axs[0, 0], axs[0, 1]],
    node_names=['node 0 - simulation', 'node 1 - simulation'],
    show=False
)
_ = qq_plots(
    residuals=residuals_list[0],
    ax=[axs[1, 0], axs[1, 1]],
    node_names=['node 0 - estimation', 'node 1 - estimation'],
    show=False
)
plt.show()
