import matplotlib.pyplot as plt
from tick.inference import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkes, HawkesKernelExp
from matplotlib import rcParams

rcParams['figure.figsize'] = (11, 7)
h = SimuHawkes(kernels=[[0, HawkesKernelExp(0.1 / 0.2, 0.2)],
                        [HawkesKernelExp(0.1 / 0.2, 0.2), 0]],
               baseline=[0.05, 0.05], seed=102983,
               verbose=False)
h.end_time = 1000000
h.simulate()

e = HawkesConditionalLaw()
e.fit(h.timestamps)
fig = plot_hawkes_kernels(e, hawkes=h, show=False)
fig.set_size_inches(12, 7)

plt.show()
