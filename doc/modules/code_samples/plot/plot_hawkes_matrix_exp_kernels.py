import numpy as np

from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesExpKern
from tick.optim.model import ModelHawkesFixedExpKernLogLik
from tick.optim.prox import ProxL1
from tick.optim.solver import AGD

end_time = 1000
n_realizations = 10

decay = 3.
baseline = [0.12, 0.07]
adjacency = [[.3, 0.], [.6, .21]]

hawkes_exp_kernels = SimuHawkesExpKernels(
    adjacency=adjacency, decays=decay, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
multi.simulate()

model = ModelHawkesFixedExpKernLogLik(decay)
model.fit(multi.timestamps)

prox = ProxL1(0.01, positive=True)

solver = AGD(step=2e-2, linesearch=False)
solver.set_model(model).set_prox(prox)
solver.solve(0.3 * np.ones(model.n_coeffs))

# Only learners can be passed to plot_hawkes_kernels...
hawkes_to_plot = HawkesExpKern(decay, max_iter=1)
hawkes_to_plot.fit(multi.timestamps[0])
hawkes_to_plot._set('coeffs', solver.solution)
plot_hawkes_kernels(hawkes_to_plot, hawkes=hawkes_exp_kernels)
