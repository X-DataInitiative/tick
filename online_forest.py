
from tick.simulation import SimuLinReg, weights_sparse_gauss
import numpy as np

n_samples = 7
n_features = 2

w0 = weights_sparse_gauss(n_features, nnz=2)
X, y = SimuLinReg(w0, -1., n_samples=n_samples).simulate()


from tick.inference import OnlineForest


forest = OnlineForest(n_trees=1)

forest.set_data(X, y)

forest.fit(n_iter=5 * n_samples)

forest.print()
