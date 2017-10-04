import numpy as np
import matplotlib.pyplot as plt

from tick.plot import stems, plot_history

from tick.optim.prox import ProxZero, ProxL2Sq

from tick.simulation import SimuPoisReg
from tick.optim.model import ModelPoisReg
from tick.optim.solver import LBFGSB, SDCA, Newton

n_samples = 10000
n_features = 40
nn_z = 1.

mask_zeros = np.random.choice(range(n_features), int((1 - nn_z) * n_features),
                              replace=True)

positive_weights = False
positive_features = False

weights = np.random.normal(size=n_features)
weights[mask_zeros] = 0

if positive_weights:
    weights = np.abs(weights)

features = np.random.randn(n_samples, n_features)
if positive_features:
    features = np.abs(features)

epsilon = 1e-1
while features.dot(weights).min() <= epsilon:
    n_fail = sum(features.dot(weights) <= epsilon)
    features[features.dot(weights) <= epsilon] = \
        np.random.randn(n_fail, n_features)

simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                   link='identity')
features, labels = simu.simulate()

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_l2sq = 1e-3

sdca = SDCA(l_l2sq)
sdca.set_model(model).set_prox(ProxZero())
sdca.solve()

lbfgsb = LBFGSB()
lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
lbfgsb.solve(0.2 * np.ones(model.n_coeffs))

newton = Newton()
newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(0.2 * np.ones(model.n_coeffs))

# stems([weights, sdca.solution, lbfgsb.solution, newton.solution],
#       titles=["Weights", 'SDCA', 'L-BFGS-B', 'Newtom'])
#
# plt.show()

plot_history([sdca, newton], log_scale=True, dist_min=True, x='time')
