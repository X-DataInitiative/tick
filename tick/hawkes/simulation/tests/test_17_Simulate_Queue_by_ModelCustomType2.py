# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
MaxN = 10
mu_i = [np.array([1.42559266,0.71285859,0.91608318,1.04369095,0.93826717,0.98857302,0.9269507, 0.87988071,0.81952085,0.8221487]),
        np.array([0.26746583,0.35055464,0.47732612,0.60000532,0.63844536,0.73559956 ,0.83674834,0.8316625, 0.84332456,0.93058838])]

beta = 100
end_time = 1000

kernels = np.array([
            [HawkesKernelExp(0.6430364, beta), HawkesKernelExp(0.06335673, beta)],
            [HawkesKernelExp(0.05711409, beta), HawkesKernelExp(0.72201187, beta)],
])

#current_num avg avg_order_size
extrainfo = np.array([20.0, 40, 5.4, -4.5])


Qty_list = []
for num_simu in range(1000):
    seed = num_simu * 10086 + 3007
    simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom='Type2', seed=seed, MaxN_of_f=MaxN, f_i=mu_i,
                            extrainfo=extrainfo, simu_mode="generate")
    for i in range(n_nodes):
        # simu_model.set_baseline(i, 0.2 + 0.1 * i)
        simu_model.set_baseline(i, 0.0)
        for j in range(n_nodes):
            simu_model.set_kernel(i, j, kernels[i, j])
    simu_model.track_intensity(0.1)

    simu_model.simulate()

    Qty = np.array(simu_model._pp.get_Qty())
    Qty_list.append(Qty)


import matplotlib.pyplot as plt
length = []
for Qty in Qty_list:
    length.append(len(Qty))
plt.hist(length, bins = 50)
plt.show()
plt.close()

import matplotlib.pyplot as plt
length = []
for Qty in Qty_list:
    if len(Qty) > 5000:
        length.append(Qty[5000])
plt.hist(length, bins=50)
plt.show()
plt.close()