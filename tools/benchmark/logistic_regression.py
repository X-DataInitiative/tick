import argparse
import time

from tick.linear_model import LogisticRegression, SimuLogReg
from tick.simulation import weights_sparse_gauss


def run(n_samples):
    n_features = 500
    weights0 = weights_sparse_gauss(n_features, nnz=100)
    intercept0 = 0.2
    X, y = SimuLogReg(weights=weights0, intercept=intercept0,
                      n_samples=n_samples, seed=123, verbose=False).simulate()

    clf_tick = LogisticRegression(C=1e4, penalty='l1', tol=0, max_iter=100)
    clf_tick.fit(X, y)


parser = argparse.ArgumentParser(description='Run dense logistic regression')
parser.add_argument('--n', metavar='n_samples', type=int, default=10000,
                    help='Number of samples')

args = parser.parse_args()

start = time.time()
run(args.n)
print(__file__, time.time() - start)
