from mlpp.simulation import SimuHawkes, HawkesKernelExp, HawkesKernelTimeFunc
import matplotlib.pyplot as plt
from pylab import rcParams
from mlpp.base.utils import TimeFunction
import numpy as np

rcParams['figure.figsize'] = 20, 8

T = np.array([0, 1, 1.5], dtype=float)
Y = np.array([0, .2, 0], dtype=float)
tf1 = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight, dt=0.1)
kernel_1 = HawkesKernelTimeFunc(tf1)

T = np.array([0, .1, 2], dtype=float)
Y = np.array([0, .4, -0.2], dtype=float)
tf2 = TimeFunction([T, Y], inter_mode=TimeFunction.InterLinear, dt=0.1)
kernel_2 = HawkesKernelTimeFunc(tf2)

hawkes = SimuHawkes(kernels=
                    [[kernel_1, kernel_1],
                     [HawkesKernelExp(.07, 4), kernel_2]],
                    baseline=[1.5, 1.5],
                    verbose=False)

hawkes.plot_kernels()

run_time = 200
dt = 0.01
hawkes.track_intensity(dt)
hawkes.end_time = run_time
hawkes.simulate()

plt.figure(figsize=(20, 8))

ax1 = plt.subplot(211)
hawkes.plot(ax=ax1, dim=0, t_max=20)

ax2 = plt.subplot(212)
hawkes.plot(ax=ax2, dim=1, t_max=20)

plt.show()
