import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

from tick.simulation import SimuHawkes, HawkesKernelPowerLaw
from tick.plot import plot_point_process

rcParams['figure.figsize'] = 20, 8

kernel_diag = HawkesKernelPowerLaw(1, 1, 3, error=1e-5)
kernel_cross = HawkesKernelPowerLaw(2, 1, 2, error=1e-5)

hawkes = SimuHawkes(kernels=[[kernel_diag, kernel_cross],
                             [0, kernel_diag]],
                    baseline=[1.5, 1.5],verbose=False, seed=2309)

run_time = 12
hawkes.track_intensity(0.01)
hawkes.end_time = 200
hawkes.simulate()

plot_point_process(hawkes, n_points=50000, t_max=8)
