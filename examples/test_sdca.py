
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from tick.linear_model import SimuLogReg, ModelLogReg
from tick.simulation import weights_sparse_gauss
from tick.solver import SDCA
from tick.prox import ProxZero

seed = 1398
np.random.seed(seed)

n_samples = 50000
n_features = 5000
sparsity = 1e-4

weights = weights_sparse_gauss(n_features, nnz=1000)
intercept = 0.2
features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')

simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                       verbose=False, intercept=intercept)
features, labels = simulator.simulate()

model = ModelLogReg(fit_intercept=True)
model.fit(features, labels)

sdca = SDCA(1e-3, verbose=True, tol=0)
sdca.set_model(model).set_prox(ProxZero())

sdca.solve()

sdca.history.print_order += ['time']
sdca.print_history()
