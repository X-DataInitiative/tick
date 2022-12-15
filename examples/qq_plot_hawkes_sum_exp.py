from tick.hawkes import SimuHawkesSumExpKernels
from tick.plot import qq_plots


def simulate_hawkes_sum_exp_kern(
        decays=[0.1, 0.5, 1.],
        baseline=[0.12, 0.07],
        adjacency=[[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]],
        end_time=1000,
        max_jumps=5000,
        verbose=True,
        force_simulation=False,
):
    model = SimuHawkesSumExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=end_time,
        max_jumps=max_jumps,
        verbose=verbose,
        force_simulation=force_simulation,
    )
    model.track_intensity(intensity_track_step=0.1)
    model.simulate()
    model.store_compensator_values()
    return model


def plot(model):
    return qq_plots(model)


def model_from_timestamps(
        timestamps,
        decays=[0.1, 0.5, 1.],
        baseline=[0.12, 0.07],
        adjacency=[[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]],
        end_time=1000,
):
    model = SimuHawkesSumExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=end_time,
    )
    model.set_timestamps(timestamps, end_time=end_time)
    model.store_compensator_values()
    return model


def round_trip(
        simu_decays=[0.1, 0.5, 1.],
        simu_baseline=[0.12, 0.07],
        simu_adjacency=[[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]],
        gfit_decays=[0.2, 0.6, 1.],
        gfit_baseline=[0.12, 0.07],
        gfit_adjacency=[[[0, .1, .8], [.2, 0.8, .2]],
                        [[0.3, 0.3, 0], [.6, .3, 0]]],
        end_time=1000,
        max_jumps=5000,
        verbose=True,
        force_simulation=False,
):
    """
    If any of `gfit_` parameters do not match with the `simu_` parameters,
    the qqplots of `gfit_model` are expected to show poor fit.
    """
    simu_model = simulate_hawkes_sum_exp_kern(
        adjacency=simu_adjacency,
        decays=simu_decays,
        baseline=simu_baseline,
        end_time=end_time,
        max_jumps=max_jumps,
        verbose=verbose,
        force_simulation=force_simulation,
    )
    gfit_model = model_from_timestamps(simu_model.timestamps,
                                       adjacency=gfit_adjacency,
                                       decays=gfit_decays,
                                       baseline=gfit_baseline,
                                       end_time=end_time,
                                       )
    return simu_model, gfit_model


def main():
    model = simulate_hawkes_sum_exp_kern()
    fig = plot(model)
    return model, fig


if __name__ == '__main__':
    main()
