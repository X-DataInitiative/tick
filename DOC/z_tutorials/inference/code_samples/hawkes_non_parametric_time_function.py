import matplotlib.pyplot as plt
import numpy as np
from tick.base import TimeFunction
from tick.inference import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkes, HawkesKernelExp, HawkesKernelTimeFunc

# A "step" kernel
T = np.array([0, 1, 1.5], dtype=float)
Y = np.array([0, .2, 0], dtype=float)
tf1 = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel_1 = HawkesKernelTimeFunc(tf1)

# Another "step" kernel
T = np.array([0, .1, 2], dtype=float)
Y = np.array([0, .1, 0], dtype=float)
tf2 = TimeFunction([T, Y], inter_mode=TimeFunction.InterLinear, dt=0.1)
kernel_2 = HawkesKernelTimeFunc(tf2)

# We put some small negative components
hawkes = SimuHawkes(kernels=[[kernel_1, HawkesKernelExp(-.03, 3), ],
                             [HawkesKernelExp(.07, 4), kernel_2]],
                    baseline=[3.5, 1.5], verbose=False, seed=13098)

hawkes.end_time = 500000
hawkes.simulate()

e = HawkesConditionalLaw(delta_lag=0.01, max_lag=10,
                         n_quad=100, max_support=3, min_support=0.002,
                         quad_method='lin', n_threads=-1)
e.fit(hawkes.timestamps)
fig = plot_hawkes_kernels(e, hawkes=hawkes, show=False)

fig.set_size_inches(12, 7)
# fig.savefig("hawkes_non_parametric_time_function.png", dpi=100)
plt.show()
