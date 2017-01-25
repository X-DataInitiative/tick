from mlpp.simulation import SimuHawkes, HawkesKernelExp
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 20, 4
run_time = 2000

hawkes = SimuHawkes(n_nodes=1, end_time=run_time,
                    verbose=False)  # 1 dimension Hawkes
kernel = HawkesKernelExp(1 / 4, 4)  # Specifying the kernel element (0,0)
hawkes.set_kernel(0, 0, kernel)
hawkes.set_baseline(0, 1.5)  # And the exogeneous intensity element 0

dt = 0.01
hawkes.track_intensity(dt)
hawkes.simulate()
ticks = hawkes.process
intensity = hawkes.tracked_intensity
intensity_times = hawkes.intensity_tracked_times

ax1 = plt.subplot(121)
hawkes.plot(ax=ax1, t_min=2, t_max=20)

ax2 = plt.subplot(122)
hawkes.plot(ax=ax2, t_min=2, max_points=25)

plt.show()
