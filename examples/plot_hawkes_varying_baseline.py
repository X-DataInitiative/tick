"""
=========================================
Hawkes process with non constant baseline
=========================================

This example simulates and then estimates Hawkes kernels with varying
baselines. In this example the intensity is written the following way

:math:`\\lambda_i(t) = \\mu_i(t) + \\sum_{j=1}^D \\int \\phi_{ij}(t - s)dN_j(s)`

Kernels are sum of exponentials and varying baseline :math:`\\mu_i(t)`
piecewise constant.
"""

import matplotlib.pyplot as plt

from tick.plot import plot_hawkes_baseline_and_kernels, qq_plots
from tick.hawkes import (SimuHawkesSumExpKernels, SimuHawkesMulti,
                         HawkesSumExpKern)


def instantiate_and_run():
    period_length = 300
    baselines = [[0.3, 0.5, 0.6, 0.4, 0.2, 0], [0.8, 0.5, 0.2, 0.3, 0.3, 0.4]]
    n_baselines = len(baselines[0])
    decays = [.5, 2., 6.]
    adjacency = [[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]]

    # simulation
    hawkes = SimuHawkesSumExpKernels(baseline=baselines,
                                     period_length=period_length,
                                     decays=decays,
                                     adjacency=adjacency,
                                     seed=2093,
                                     verbose=False,
                                     )
    hawkes.end_time = 1000
    hawkes.adjust_spectral_radius(0.5)

    multi = SimuHawkesMulti(hawkes, n_simulations=4)
    multi.simulate()

    # estimation
    learner = HawkesSumExpKern(decays=decays, n_baselines=n_baselines,
                               period_length=period_length)

    learner.fit(multi.timestamps)
    return hawkes, learner


def plot(learner, hawkes, show=True):
    # plot
    fig = plot_hawkes_baseline_and_kernels(learner, hawkes=hawkes, show=False)
    if show:
        fig.tight_layout()
        plt.show()
    return fig


def simulated_v_estimated_qq_plots(
        model,
        learner,
        show=True,
):
    fig, ax_list = plt.subplots(2, 1, figsize=(10, 6))
    timestamps = model.timestamps
    end_time = model.end_time
    learner.qq_plots(events=timestamps, end_time=end_time, ax=ax_list)
    model.store_compensator_values()
    qq_plots(model, ax=ax_list)

    # Enhance plot
    for ax in ax_list:
        # Set labels to both plots
        ax.lines[0].set_label('estimated')
        ax.lines[2].set_label('original')

        # Change original intensity style
        ax.lines[2].set_alpha(0.6)
        ax.lines[3].set_alpha(0.6)

        ax.lines[2].set_markerfacecolor('orange')
        ax.lines[2].set_markeredgecolor('orange')

        ax.legend()

    if show:
        fig.tight_layout()
        plt.show()
    return fig


def main():
    hawkes, learner = instantiate_and_run()
    fig = plot(learner, hawkes, show=True)
    return fig


if __name__ == '__main__':
    main()
