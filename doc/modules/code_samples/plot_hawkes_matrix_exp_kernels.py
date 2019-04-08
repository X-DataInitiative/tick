from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti, HawkesExpKern
import matplotlib.pyplot as plt

end_time = 1000
n_realizations = 10

decays = [[4., 1.], [2., 2.]]
baseline = [0.12, 0.07]
adjacency = [[.3, 0.], [.6, .21]]

hawkes_exp_kernels = SimuHawkesExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
multi.simulate()

learner = HawkesExpKern(decays, penalty='l1', C=10)
learner.fit(multi.timestamps)

plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels)