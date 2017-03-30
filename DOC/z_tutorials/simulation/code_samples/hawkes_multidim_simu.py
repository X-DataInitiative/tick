import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

from tick.simulation import SimuHawkesExpKernels
from tick.plot import plot_point_process

rcParams['figure.figsize'] = 20, 8

n_nodes = 3  # dimension of the Hawkes process
adjacency = 0.2 * np.ones((n_nodes, n_nodes))
adjacency[0, 1] = 0
decays = 3 * np.ones((n_nodes, n_nodes))
baseline = 0.5 * np.ones(n_nodes)
hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays,
                              baseline=baseline, verbose=False, seed=2398)

run_time = 100
hawkes.end_time = run_time
dt = 0.01
hawkes.track_intensity(dt)
hawkes.simulate()

plot_point_process(hawkes, n_points=50000, t_min=10, max_jumps=30)
