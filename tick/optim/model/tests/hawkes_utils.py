from scipy.integrate import quad
import numpy as np


def hawkes_least_square_error(intensities, timestamps, end_time, precision=3):
    dim = len(timestamps)

    squared_intensity_integral = sum(
        [quad(lambda x, i=i: intensities[i](x) ** 2, 0, end_time,
              epsabs=np.power(10., -precision), limit=1000)[0]
         for i in range(dim)]
    )

    intensity_convolution = sum(
        [sum([intensities[i](t) for t in timestamps[i]])
         for i in range(dim)]
    )

    return squared_intensity_integral - 2 * intensity_convolution


def hawkes_log_likelihood(intensities, timestamps, end_time, precision=3):
    dim = len(timestamps)

    log_intensity = sum(
        [quad(lambda x, i=i: intensities[i](x), 0, end_time,
              epsabs=np.power(10., -precision), limit=1000)[0]
         for i in range(dim)]
    )

    intensity_integral = sum(
        [sum([np.log(intensities[i](t)) for t in timestamps[i]])
         for i in range(dim)]
    )

    return log_intensity - intensity_integral


def hawkes_exp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)

    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: \
                adjacency[i, j] * decays[i, j] * np.exp(- decays[i, j] * t)

    intensities = {}
    for i in range(dim):
        intensities[i] = lambda x, i=i: baseline[i] + sum(
            [sum(kernels[i][j](x - timestamps[j][timestamps[j] < x]))
             for j in range(dim)]
        )

    return intensities


def hawkes_sumexp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)

    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: \
                sum([alpha * beta * np.exp(- beta * t)
                     for alpha, beta in zip(adjacency[i, j], decays)]
                    )

    intensities = {}
    for i in range(dim):
        intensities[i] = lambda x, i=i: baseline[i] + sum(
            [sum(kernels[i][j](x - timestamps[j][timestamps[j] < x]))
             for j in range(dim)]
        )

    return intensities
