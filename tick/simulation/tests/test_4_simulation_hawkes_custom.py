# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation.build.simulation import Hawkes

n_nodes = 2
seed = 10086
MaxN_of_f = 5
f_i = [np.array([1.0, 7, 7.7, 6, 3]), np.array([1.0, 0.5, 2, 1, 2])]

simu_model = Hawkes_custom(n_nodes, seed, MaxN_of_f, f_i)
# print(simu_model.simulate)
simu_model.simulate(10000)


print(simu_model.get_timestamps())