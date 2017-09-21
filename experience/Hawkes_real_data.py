import numpy as np
import matplotlib.pyplot as plt

from tick.dataset import fetch_hawkes_bund_data
from tick.inference import HawkesConditionalLaw, HawkesDual, HawkesSumExpKern
from tick.plot import plot_hawkes_kernel_norms

timestamps_list = fetch_hawkes_bund_data()
l_l2sq = 1e0
decays = np.logspace(0, 4.2, 10)

# Train

# Wiener Hopf
kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))

hawkes_learner = HawkesConditionalLaw(
    claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
    quad_method="log", n_quad=10, min_support=1e-4, max_support=1,
    n_threads=4)

hawkes_learner.fit(timestamps_list)

# Dual
hawkes_dual = HawkesDual(decays, l_l2sq, n_threads=4, verbose=True,
                         max_iter=100)
hawkes_dual.fit(timestamps_list)

# LBFGS
hawkes_lbgfsb = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                 C=1 / l_l2sq, penalty='l2', solver='l-bfgs-b',
                                 record_every=1)
hawkes_lbgfsb._model_obj.n_threads = 4
hawkes_lbgfsb.fit(timestamps_list)

# Plots
fig, ax_list = plt.subplots(1, 3, figsize=(16, 4))
plot_hawkes_kernel_norms(hawkes_learner,
                         node_names=["P_u", "P_d", "T_a", "T_b"],
                         ax=ax_list[0])
ax_list[0].set_xlabel('Wiener Hopf', fontsize=20)

plot_hawkes_kernel_norms(hawkes_dual,
                         node_names=["P_u", "P_d", "T_a", "T_b"],
                         ax=ax_list[1])
ax_list[1].set_xlabel('Dual', fontsize=20)

plot_hawkes_kernel_norms(hawkes_lbgfsb,
                         node_names=["P_u", "P_d", "T_a", "T_b"],
                         ax=ax_list[2])
ax_list[2].set_xlabel('L-BFGS-B', fontsize=20)

fig.tight_layout()

plt.savefig('plot_hawkes_bund_data.png')
