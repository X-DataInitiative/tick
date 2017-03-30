import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from tick.base import TimeFunction
from tick.plot import plot_timefunction

rcParams['figure.figsize'] = 20, 4

T = np.array([0, 3, 5.9, 8.001], dtype=float)
Y = np.array([2, 4.1, 1, 2], dtype=float)

tf_1 = TimeFunction((T, Y), dt=1.2)
tf_2 = TimeFunction((T, Y), border_type=TimeFunction.BorderContinue,
                    inter_mode=TimeFunction.InterConstRight, dt=0.01)
tf_3 = TimeFunction((T, Y), border_type=TimeFunction.BorderConstant,
                    inter_mode=TimeFunction.InterConstLeft, border_value=3)

ax1 = plt.subplot(131)
plot_timefunction(tf_1, ax=ax1)
ax1.set_ylim([-0.5, 6.0])

ax2 = plt.subplot(132)
plot_timefunction(tf_2, ax=ax2)
ax2.set_ylim([-0.5, 6.0])

ax3 = plt.subplot(133)
plot_timefunction(tf_3, ax=ax3)
ax3.set_ylim([-0.5, 6.0])

plt.show()

print(ax1.__class__)