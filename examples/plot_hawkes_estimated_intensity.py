"""
============================================
Plot estimated intensity of Hawkes processes
============================================

This examples shows how the estimated intensity of a learned Hawkes process
can be plotted. In this example, the data has been generated so we are able
to compare this estimated intensity with the true intensity that has generated
the process.
"""

import matplotlib.pyplot as plt

from tick.hawkes import SimuHawkesSumExpKernels, HawkesSumExpKern
from tick.plot import plot_point_process

end_time = 1000

decays = [0.1, 0.5, 1.]
baseline = [0.12, 0.07]
adjacency = [[[0, .1, .4], [.2, 0., .2]],
             [[0, 0, 0], [.6, .3, 0]]]

hawkes_exp_kernels = SimuHawkesSumExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

hawkes_exp_kernels.track_intensity(0.1)
hawkes_exp_kernels.simulate()

learner = HawkesSumExpKern(decays, penalty='elasticnet',
                           elastic_net_ratio=0.8)
learner.fit(hawkes_exp_kernels.timestamps)

t_min = 100
t_max = 200
fig, ax_list = plt.subplots(2, 1, figsize=(10, 6))
learner.plot_estimated_intensity(hawkes_exp_kernels.timestamps,
                                 t_min=t_min, t_max=t_max,
                                 ax=ax_list)

plot_point_process(hawkes_exp_kernels, plot_intensity=True,
                   t_min=t_min, t_max=t_max, ax=ax_list)

# Enhance plot
for ax in ax_list:
    # Set labels to both plots
    ax.lines[0].set_label('estimated')
    ax.lines[1].set_label('original')

    # Change original intensity style
    ax.lines[1].set_linestyle('--')
    ax.lines[1].set_alpha(0.8)

    # avoid duplication of scatter plots of events
    ax.collections[1].set_alpha(0)

    ax.legend()

fig.tight_layout()
plt.show()
