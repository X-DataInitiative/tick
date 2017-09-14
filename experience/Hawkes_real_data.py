import numpy as np
import matplotlib.pyplot as plt

from tick.dataset import fetch_hawkes_bund_data
from tick.inference import HawkesConditionalLaw, HawkesDual
from tick.plot import plot_hawkes_kernel_norms

timestamps_list = fetch_hawkes_bund_data()

kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))

hawkes_learner = HawkesConditionalLaw(
    claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
    quad_method="log", n_quad=10, min_support=1e-4, max_support=1,
    n_threads=4)

hawkes_learner.fit(timestamps_list)

l_l2sq = 1e0
decays = np.logspace(0, 4.2, 10)
hawkes_dual = HawkesDual(decays, l_l2sq, n_threads=4, verbose=True,
                         max_iter=100)
hawkes_dual.fit(timestamps_list)

fig, ax_list = plt.subplots(1, 2, figsize=(10, 4))
plot_hawkes_kernel_norms(hawkes_learner,
                         node_names=["P_u", "P_d", "T_a", "T_b"],
                         ax=ax_list[0])
ax_list[0].set_xlabel('Wiener Hopf', fontsize=20)

plot_hawkes_kernel_norms(hawkes_dual,
                         node_names=["P_u", "P_d", "T_a", "T_b"],
                         ax=ax_list[1])
ax_list[1].set_xlabel('Dual', fontsize=20)

fig.tight_layout()

plt.savefig('plot_hawkes_bund_data.png')
