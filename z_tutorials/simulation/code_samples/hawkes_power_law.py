from mlpp.simulation import SimuHawkes, HawkesKernelPowerLaw
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 20, 8

kernel_diag = HawkesKernelPowerLaw(1, 1, 3, error=1e-5)
kernel_cross = HawkesKernelPowerLaw(2, 1, 2, error=1e-5)

hawkes = SimuHawkes(kernels=
                    [[kernel_diag, kernel_cross],
                     [0, kernel_diag]],
                    baseline=[1.5, 1.5],
                    verbose=False)

run_time = 200
dt = 0.01
hawkes.track_intensity(dt)
hawkes.end_time = run_time
hawkes.simulate()

ax1 = plt.subplot(211)
hawkes.plot(ax=ax1, dim=0, t_max=8)

ax2 = plt.subplot(212)
hawkes.plot(ax=ax2, dim=1, t_max=8)

plt.show()
