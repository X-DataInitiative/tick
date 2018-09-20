import argparse
import time

import numpy as np

from tick.linear_model import ModelPoisReg
from tick.linear_model import SimuPoisReg
from tick.prox import ProxZero
from tick.solver import SDCA


def run(n_samples):
    np.random.seed(2380)

    n_features = 500
    weights = np.random.normal(size=n_features)
    features = np.random.randn(n_samples, n_features)

    # Ensure feasible set is not empty
    epsilon = 1e-1
    while features.dot(weights).min() <= epsilon:
        n_fail = sum(features.dot(weights) <= epsilon)
        features[features.dot(weights) <= epsilon] = \
            np.random.randn(n_fail, n_features)

    simu = SimuPoisReg(weights=weights, features=features, n_samples=n_samples,
                       seed=123, verbose=False, link='identity')
    features, labels = simu.simulate()

    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    l_l2sq = 1e-2

    sdca = SDCA(l_l2sq, max_iter=1000, verbose=False)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()


parser = argparse.ArgumentParser(description='Run Poisson regression with SDCA')
parser.add_argument('--n', metavar='n_samples', type=int, default=1000,
                    help='Number of samples')

args = parser.parse_args()

start = time.time()
run(args.n)
print(__file__, time.time() - start)
