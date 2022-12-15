from tick.hawkes import SimuHawkesExpKernels
from tick.plot import qq_plots


def simulate_hawkes_exp_kern(
        decays=[[1., 1.5], [0.1, 0.5]],
        baseline=[0.12, 0.07],
        adjacency=[[.1, .4], [.2, 0.5]],
        end_time=3000,
        max_jumps=1000,
        verbose=True,
        force_simulation=False,
):
    model = SimuHawkesExpKernels(
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


def main():
    model = simulate_hawkes_exp_kern()
    fig = plot(model)
    return model, fig


if __name__ == '__main__':
    main()
