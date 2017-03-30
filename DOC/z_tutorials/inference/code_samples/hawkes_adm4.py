from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesADM4

end_time = 1000
n_nodes = 2
n_realizations = 10
timestamps_list = []

decay = 3.
baseline = [0.12, 0.07]
adjacency = [[.3, .6], [.0, .21]]

hawkes_exp_kernels = SimuHawkesExpKernels(
    adjacency=adjacency, decays=decay, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
multi.simulate()

learner = HawkesADM4(decay)
learner.fit(multi.timestamps)

plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels)
