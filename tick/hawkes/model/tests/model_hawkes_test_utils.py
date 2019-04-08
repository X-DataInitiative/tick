# License: BSD 3 clause

import numpy as np
from scipy.integrate import quad


def exponential_kernel(t, intensity, decay):
    return intensity * decay * np.exp(-decay * t)


def sum_exponential_kernel(t, intensities, decays):
    return sum([
        alpha * beta * np.exp(-beta * t)
        for alpha, beta in zip(intensities, decays)
    ])


def hawkes_intensities(timestamps, baseline, kernels):
    dim = len(baseline)
    intensities = {}
    for i in range(dim):
        intensities[i] = lambda x, i=i: baseline[i] + sum(
            [sum(kernels[i][j](x - timestamps[j][timestamps[j] < x]))
             for j in range(dim)]
        )
    return intensities


def hawkes_intensities_varying_baseline(timestamps, baseline, kernels):
    # in this case baseline is a function of time
    dim = len(baseline)
    intensities = {}
    for i in range(dim):
        intensities[i] = lambda x, i=i: baseline[i](x) + sum(
            [sum(kernels[i][j](x - timestamps[j][timestamps[j] < x]))
             for j in range(dim)]
        )
    return intensities


def hawkes_least_square_error(intensities, timestamps, end_time, precision=3):
    dim = len(timestamps)

    squared_intensity_integral = sum([
        quad(lambda x, i=i: intensities[i](x) ** 2, 0, end_time,
             epsabs=np.power(10., -precision), limit=1000)[0]
        for i in range(dim)
    ])

    intensity_convolution = sum(
        [sum([intensities[i](t) for t in timestamps[i]]) for i in range(dim)])

    return squared_intensity_integral - 2 * intensity_convolution


def hawkes_log_likelihood(intensities, timestamps, end_time, precision=3):
    dim = len(timestamps)

    compensator = sum([
        quad(lambda x, i=i: intensities[i](x), 0, end_time, epsabs=np.power(
            10., -precision), limit=1000)[0] for i in range(dim)
    ])

    log_intensity = sum([
        sum([np.log(intensities[i](t)) for t in timestamps[i]])
        for i in range(dim)
    ])

    return dim * end_time - compensator + log_intensity


def hawkes_exp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)

    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: \
                exponential_kernel(t, adjacency[i, j], decays[i, j])

    return hawkes_intensities(timestamps, baseline, kernels)


def hawkes_sumexp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)

    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: \
                sum_exponential_kernel(t, adjacency[i, j], decays)

    return hawkes_intensities(timestamps, baseline, kernels)


def hawkes_sumexp_kernel_varying_intensities(baseline, decays, adjacency,
                                             timestamps):
    # in this case baseline is a function of time
    dim = len(timestamps)

    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: \
                sum_exponential_kernel(t, adjacency[i, j], decays)

    return hawkes_intensities_varying_baseline(timestamps, baseline, kernels)
