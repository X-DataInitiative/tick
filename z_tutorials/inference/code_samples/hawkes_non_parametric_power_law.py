import numpy as np
import matplotlib.pyplot as plt
from mlpp.hawkesnoparam.estim import Estim
from mlpp.simulation import SimuHawkes, HawkesKernelPowerLaw

np.set_printoptions(precision=4)

alphas = np.array([0.012, 0.008, 0.004, 0.0005])
delta = 0.0005
beta = 1.3

hawkes = SimuHawkes(kernels=[[HawkesKernelPowerLaw(alphas[0], delta, beta, 2000),
                              HawkesKernelPowerLaw(alphas[1], delta, beta, 2000)],
                             [HawkesKernelPowerLaw(alphas[2], delta, beta, 2000),
                          HawkesKernelPowerLaw(alphas[3], delta, beta, 2000)]],
                    baseline=[0.05, 0.05],
                    verbose=False)
hawkes.end_time = 50000
hawkes.simulate()

e = Estim(hDelta=0.1, hMax=100, hMin=0.002, claw_method="log")
e.add_realization(hawkes)
e.compute(n_quad=50, xmax=2000, xmin=0.002, method="log")

fig, ax_list_list = e.plot(loglogscale=True)
fig.set_size_inches(12, 7)

for ax_list in ax_list_list:
    for ax in ax_list:
        ax.legend(loc=3)

plt.show()
