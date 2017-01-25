from mlpp.simulation import SimuHawkesExpKernels
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 20, 8

d = 3  # dimension of the Hawkes process
alphas = 0.2 * np.ones((d, d))
alphas[0, 1] = 0
betas = 3 * np.ones((d, d))
mus = 0.5 * np.ones(d)
hawkes = SimuHawkesExpKernels(adjacency=alphas, decays=betas,
                              baselines=mus, verbose=False)

run_time = 1000
hawkes.end_time = run_time
dt = 0.001
hawkes.track_intensity(dt)
hawkes.simulate()

ax1 = plt.subplot(211)
hawkes.plot(ax=ax1, dim=0, t_min=300, t_max=310)

ax2 = plt.subplot(212)
hawkes.plot(ax=ax2, dim=1, max_points=28, t_max=310)

plt.show()
