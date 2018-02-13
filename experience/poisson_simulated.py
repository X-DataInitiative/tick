import numpy as np
import matplotlib.pyplot as plt

from tick.plot import stems, plot_history

from tick.optim.prox import ProxZero, ProxL2Sq

from tick.simulation import SimuPoisReg
from tick.optim.model import ModelPoisReg
from tick.optim.solver import LBFGSB, SDCA, Newton

n_samples = 10000
n_features = 5

positive_weights = False
positive_features = True

weights = np.random.normal(size=n_features)


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


print(features.dot(weights).min())
features = (features.T / np.linalg.norm(features, axis=1)).T
print(features.dot(weights).min())

simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                   link='identity')
features, labels = simu.simulate()

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_l2sq = 1e-1

sdca = SDCA(l_l2sq, max_iter=100, print_every=10, tol=1e-16)
sdca.set_model(model).set_prox(ProxZero())
sdca.solve()

newton = Newton()
newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(0.2 * np.ones(model.n_coeffs))

fig, axes = plt.subplots(2, 1, figsize=(4, 6))

plot_history([sdca, newton], log_scale=True, dist_min=True, x='time', ax=axes[0])

ax = axes[1]
print(max(sdca.dual_solution))
ax.hist(sdca.dual_solution, bins=30)

plt.show()
