"""
=====================================
Hawkes simulation with exotic kernels
=====================================

Simulation of Hawkes processes with usage of custom kernels
"""

import matplotlib.pyplot as plt
import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, HawkesKernelExp, HawkesKernelTimeFunc
from tick.plot import plot_point_process

t_values = np.array([0, 1, 1.5], dtype=float)
y_values = np.array([0, .2, 0], dtype=float)
tf1 = TimeFunction([t_values, y_values],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel_1 = HawkesKernelTimeFunc(tf1)

t_values = np.array([0, .1, 2], dtype=float)
y_values = np.array([0, .4, -0.2], dtype=float)
tf2 = TimeFunction([t_values, y_values],
                   inter_mode=TimeFunction.InterLinear, dt=0.1)
kernel_2 = HawkesKernelTimeFunc(tf2)

hawkes = SimuHawkes(kernels=[[kernel_1, kernel_1],
                             [HawkesKernelExp(.07, 4), kernel_2]],
                    baseline=[1.5, 1.5], verbose=False, seed=23983)

run_time = 40
dt = 0.01
hawkes.track_intensity(dt)
hawkes.end_time = run_time
hawkes.simulate()

fig, ax = plt.subplots(hawkes.n_nodes, 1, figsize=(14, 8))
plot_point_process(hawkes, t_max=20, ax=ax)

plt.show()
