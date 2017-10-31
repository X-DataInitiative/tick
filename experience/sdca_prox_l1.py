import numpy as np

from tick.optim.model import ModelLogReg, ModelPoisReg
from tick.optim.prox import ProxL1, ProxElasticNet
from tick.optim.solver import SDCA, SVRG
from tick.plot import plot_history
from tick.simulation import weights_sparse_gauss, SimuLogReg, SimuPoisReg

n_samples, n_features, = 5000, 5
weights0 = weights_sparse_gauss(n_features, nnz=3)


def get_model(model_type):
    intercept0 = None
    if model_type == 'logistic':
        X, y = SimuLogReg(weights=weights0, intercept=intercept0,
                          n_samples=n_samples, seed=123).simulate()
        model = ModelLogReg(fit_intercept=False).fit(X, y)
    else:
        features = np.random.randn(n_samples, n_features)

        epsilon = 1e-1
        while features.dot(weights0).min() <= epsilon:
            n_fail = sum(features.dot(weights0) <= epsilon)
            features[features.dot(weights0) <= epsilon] = \
                np.random.randn(n_fail, n_features)

        X, y = SimuPoisReg(weights=weights0, intercept=intercept0,
                           features=features, seed=123,
                           n_samples=n_samples, link='identity').simulate()
        model = ModelPoisReg(fit_intercept=False, link='identity').fit(X, y)

    return model

model = get_model('poisson')
ratio = 0.8
l_enet = 1e-1

# SDCA "elastic-net" formulation is different from elastic-net
# implementation
l_l1_sdca = ratio * l_enet
l_l2_sdca = (1 - ratio) * l_enet
sdca = SDCA(l_l2sq=l_l2_sdca, max_iter=100, verbose=True, tol=0).set_model(model)
prox_l1 = ProxL1(l_l1_sdca)
sdca.set_prox(prox_l1)
coeffs_sdca = sdca.solve()

# Compare with SVRG
svrg = SVRG(max_iter=100, verbose=True, tol=0).set_model(model)
prox_enet = ProxElasticNet(l_enet, ratio)
svrg.set_prox(prox_enet)
coeffs_svrg = svrg.solve(step=0.1)

print(weights0)
print(coeffs_sdca)
print(coeffs_svrg)

plot_history([sdca, svrg], log_scale=True, dist_min=True)
