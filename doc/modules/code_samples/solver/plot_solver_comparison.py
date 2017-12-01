import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss
from tick.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.model import ModelLogReg
from tick.prox import ProxElasticNet, ProxL1
from tick.plot import plot_history

n_samples, n_features, = 5000, 50
weights0 = weights_sparse_gauss(n_features, nnz=10)
intercept0 = 0.2
X, y = SimuLogReg(weights=weights0, intercept=intercept0,
                  n_samples=n_samples, seed=123, verbose=False).simulate()

model = ModelLogReg(fit_intercept=True).fit(X, y)
prox = ProxElasticNet(strength=1e-3, ratio=0.5, range=(0, n_features))

solver_params = {'max_iter': 100, 'tol': 0., 'verbose': False}
x0 = np.zeros(model.n_coeffs)

gd = GD(linesearch=False, **solver_params).set_model(model).set_prox(prox)
gd.solve(x0, step=1 / model.get_lip_best())

agd = AGD(linesearch=False, **solver_params).set_model(model).set_prox(prox)
agd.solve(x0, step=1 / model.get_lip_best())

sgd = SGD(**solver_params).set_model(model).set_prox(prox)
sgd.solve(x0, step=500.)

svrg = SVRG(**solver_params).set_model(model).set_prox(prox)
svrg.solve(x0, step=1 / model.get_lip_max())

plot_history([gd, agd, sgd, svrg], log_scale=True, dist_min=True)
