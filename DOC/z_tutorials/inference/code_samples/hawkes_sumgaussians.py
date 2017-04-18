import numpy as np

from tick.plot import plot_hawkes_kernels, plot_point_process
from tick.simulation import (SimuHawkes, SimuHawkesMulti,
                             HawkesKernelExp, HawkesKernelTimeFunc,
                             HawkesKernelPowerLaw, HawkesKernel0)
from tick.inference import HawkesSumGaussians

end_time = 1000
n_nodes = 2
n_realizations = 10
n_gaussians = 5

timestamps_list = []

kernel_timefunction = HawkesKernelTimeFunc(
    t_values=np.array([0., .7, 2.5, 3., 4.]),
    y_values=np.array([.3, .03, .03, .2, 0.])
)
kernels = [[HawkesKernelExp(.2, 2.), HawkesKernelPowerLaw(.2, .5, 1.3)],
           [HawkesKernel0(), kernel_timefunction]]

hawkes = SimuHawkes(baseline=[.5, .2], kernels=kernels,
                    end_time=end_time, verbose=False, seed=1039)

multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations)

multi.simulate()

learner = HawkesSumGaussians(n_gaussians, max_iter=10)
learner.fit(multi.timestamps)

plot_hawkes_kernels(learner, hawkes=hawkes, support=4)
