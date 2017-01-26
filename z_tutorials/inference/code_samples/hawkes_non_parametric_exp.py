import matplotlib.pyplot as plt
from mlpp.hawkesnoparam.estim import Estim
from mlpp.simulation import SimuHawkes, HawkesKernelExp

h = SimuHawkes(kernels=[[0, HawkesKernelExp(0.1 / 0.2, 0.2)],
                        [HawkesKernelExp(0.1 / 0.2, 0.2), 0]],
               baseline=[0.05, 0.05],
               verbose=False)
h.end_time = 1000000
h.simulate()

e = Estim(h)
e.compute()
fig, _ = e.plot()
fig.set_size_inches(12, 7)

plt.show()
