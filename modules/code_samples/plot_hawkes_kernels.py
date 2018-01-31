import numpy as np
import matplotlib.pyplot as plt

from tick.hawkes import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelTimeFunc

kernel_0 = HawkesKernel0()
kernel_exp = HawkesKernelExp(.7, 1.3)
kernel_pl = HawkesKernelPowerLaw(.1, .2, 0.7)

t_values = np.array([0, 1, 1.5, 1.8, 2.7])
y_values = np.array([0, .6, .34, .2, .1])
kernel_tf = HawkesKernelTimeFunc(t_values=t_values, y_values=y_values)

kernels = [[kernel_0, kernel_exp], [kernel_pl, kernel_tf]]

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 4))

t_values = np.linspace(0, 3, 100)
for i in range(2):
    for j in range(2):
        ax[i, j].plot(t_values, kernels[i][j].get_values(t_values),
                      label=kernels[i][j])
        ax[i, j].legend()
plt.show()