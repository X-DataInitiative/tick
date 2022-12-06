"""
================================================
Plot estimated intensity of Hawkes processes and
assess goodness of fit via QQ plots
=================================================

This examples shows how the estimated intensity of a learned Hawkes process
can be plotted. In this example, the data has been generated so we are able
to compare this estimated intensity with the true intensity that has generated
the process.
"""

import matplotlib.pyplot as plt

from tick.hawkes import (
    # SimuHawkesExpKernels,
    # HawkesExpKern,
    SimuHawkesSumExpKernels,
    HawkesSumExpKern
)
from tick.plot import plot_point_process, qq_plots


def simulate(
    Simulator=SimuHawkesSumExpKernels,
    end_time=1000,
    decays=[0.1, 0.5, 1.],
    baseline=[0.12, 0.07],
    adjacency=[[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]],
):
    model = Simulator(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=end_time,
        verbose=False,
        seed=1039,
    )
    model.track_intensity(0.1)
    model.simulate()
    return model


def fit(
        timestamps,
        Fitter=HawkesSumExpKern,
        decays=[0.1, 0.5, 1.],
        **kwargs,
):
    if Fitter == HawkesSumExpKern:
        if 'penalty' not in kwargs:
            kwargs['penalty'] = 'elasticnet'
            kwargs['elastic_net_ratio'] = 0.8
    learner = Fitter(decays=decays, **kwargs)
    learner.fit(timestamps)
    return learner


def plot_intensities(
    model,
    learner,
    t_min=100,
    t_max=200,
    show=True,
):
    fig, ax_list = plt.subplots(2, 1, figsize=(10, 6))
    learner.plot_estimated_intensity(model.timestamps, t_min=t_min,
                                     t_max=t_max, ax=ax_list)
    plot_point_process(model, plot_intensity=True, t_min=t_min,
                       t_max=t_max, ax=ax_list)

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


def main(
        Simulator=SimuHawkesSumExpKernels,
        Fitter=HawkesSumExpKern,
        simu_decays=[0.1, 0.5, 1.],
        fit_decays=[0.1, 0.5, 1.],
        goodness_of_fit=False,

):
    model = simulate(Simulator=Simulator, decays=simu_decays)
    learner = fit(model.timestamps, Fitter=Fitter, decays=fit_decays)
    fig_intensities = plot_intensities(model, learner, show=False)
    fig_intensities.tight_layout()
    if goodness_of_fit:
        fig_gfit = simulated_v_estimated_qq_plots(model, learner, show=False)
        fig_gfit.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
