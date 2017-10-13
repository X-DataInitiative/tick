import numpy as np
import matplotlib.pyplot as plt

from tick.dataset import fetch_hawkes_bund_data
from tick.inference import HawkesConditionalLaw, HawkesDual, HawkesSumExpKern
from tick.optim.prox import ProxL2Sq
from tick.optim.solver import Newton
from tick.plot import plot_hawkes_kernel_norms

timestamps_list = fetch_hawkes_bund_data()

train_timestamps = timestamps_list[:3]
l_l2sq = 1e0
decays = np.logspace(0, 4.2, 10)

# Train


max_iter = 10
# Dual
hawkes_dual = HawkesDual(decays, l_l2sq, n_threads=4, verbose=True,
                         max_iter=max_iter)
hawkes_dual.fit(train_timestamps)
dual_coeffs = hawkes_dual.coeffs

# LBFGS
hawkes_lbgfsb = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                 C=1 / l_l2sq, penalty='l2', solver='l-bfgs-b',
                                 record_every=1, max_iter=max_iter)
hawkes_lbgfsb._model_obj.n_threads = 4
hawkes_lbgfsb.fit(train_timestamps)

model = hawkes_lbgfsb._construct_model_obj()
model.fit(train_timestamps)
newton = Newton(max_iter=1).set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(dual_coeffs)


print(hawkes_dual.score(timestamps_list[2]))
print(hawkes_lbgfsb.score(timestamps_list[2]))


# # Plots
# fig, ax_list = plt.subplots(1, 2, figsize=(16, 4))
#
# plot_hawkes_kernel_norms(hawkes_dual,
#                          node_names=["P_u", "P_d", "T_a", "T_b"],
#                          ax=ax_list[0])
# ax_list[2].set_xlabel('Dual', fontsize=20)
#
# plot_hawkes_kernel_norms(hawkes_lbgfsb,
#                          node_names=["P_u", "P_d", "T_a", "T_b"],
#                          ax=ax_list[1])
# ax_list[1].set_xlabel('L-BFGS-B', fontsize=20)
#
# fig.tight_layout()
#
# plt.show()