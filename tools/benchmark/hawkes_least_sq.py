import argparse
import time

import numpy as np

from tick.hawkes.simulation import SimuHawkesSumExpKernels, SimuHawkesMulti
from tick.hawkes.model import ModelHawkesSumExpKernLeastSq

def run(n_nodes, n_decays, end_time, n_threads, n_simulations, seed, verbose):
    np.random.seed(seed)

    baseline = 1 + np.random.rand(n_nodes)
    decays = np.random.rand(n_decays)
    adjacency = np.random.rand(n_nodes, n_nodes, n_decays)

    # simulation
    hawkes = SimuHawkesSumExpKernels(baseline=baseline, decays=decays,
                                     adjacency=adjacency, seed=seed,
                                     verbose=False)
    hawkes.end_time = end_time
    hawkes.adjust_spectral_radius(0.3)

    multi = SimuHawkesMulti(hawkes, n_simulations=n_simulations,
                            n_threads=n_threads)
    if verbose:
        start = time.time()
    multi.simulate()
    if verbose:
        print(__file__, "simulation", time.time() - start)

    # estimation
    model = ModelHawkesSumExpKernLeastSq(decays, n_threads=n_threads)
    model.fit(multi.timestamps)

    if verbose:
        start = time.time()
    for _ in range(100):
        model._model.compute_weights()
    if verbose:
        print(__file__, "compute weights", time.time() - start)


parser = argparse.ArgumentParser(description='Run weights computations for'
                                             'Hawkes least square model')
parser.add_argument('--d', metavar='n_nodes', type=int, default=5,
                    help='Number of nodes of the Hawkes model. Time should '
                         'scale linearly with the square of this parameter')
parser.add_argument('--u', metavar='n_decays', type=int, default=3,
                    help='Number of decays of the Hawkes model. Time should '
                         'scale linearly with this parameter')
parser.add_argument('--e', metavar='end_time', type=int, default=1000,
                    help='Time until which the Hawkes model is simulated. '
                         'Time should scale linearly with this parameter')
parser.add_argument('--t', metavar='n_threads', type=int, default=1,
                    help='Number of threads used for the simulation and the '
                         'weights computation')
parser.add_argument('--n', metavar='n_simulations', type=int, default=1,
                    help='Number of different simulations of the same process'
                         'Time should scale linearly with this parameter')
parser.add_argument('--s', metavar='seed', type=int, default=1,
                    help='Seed used for Hawkes simulation. '
                         'Time should be independant with this parameter')
parser.add_argument('--p', metavar='print', type=bool, default=False,
                    help='Print first simulation time and then compute weights'
                         ' time')

args = parser.parse_args()

start = time.time()
run(args.d, args.u, args.e, args.t, args.n, args.s, args.p)

if not args.p:
    print(__file__, time.time() - start)
