import numpy as np

from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkes, HawkesKernelSumExp, HawkesKernelExp
from tick.simulation.hawkes_multi import SimuHawkesMulti
from tick.inference import HawkesEM

hawkes = SimuHawkes(end_time=30000, verbose=False, seed=203,
                    baseline=[0.2, 0.4])
hawkes.set_kernel(0, 0, HawkesKernelExp(0.05, 6))
hawkes.set_kernel(1, 0, HawkesKernelExp(0.1, 2))
hawkes.set_kernel(1, 1, HawkesKernelSumExp([0.07, 0.03], [.5, 6.]))

hawkes_multi = SimuHawkesMulti(hawkes, n_simulations=4, n_threads=4)
hawkes_multi.simulate()

kernel_discretization = np.array([0, .05, .1, .2, .5, .8, 1.3, 2., 3., 4.])
em = HawkesEM(kernel_discretization=kernel_discretization, n_threads=8,
              max_iter=30)

em.fit(hawkes_multi.timestamps)

plot_hawkes_kernels(em, hawkes=hawkes)
