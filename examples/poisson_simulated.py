import numpy as np
import matplotlib.pyplot as plt
import scipy

from tick.plot import stems, plot_history

from tick.prox import ProxZero, ProxL2Sq, ProxL1

from tick.linear_model import ModelPoisReg, SimuPoisReg
from tick.solver import SDCA
from tick.solver.newton import Newton

np.random.seed(103903)
n_samples = 1000
n_features = 6

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

link = "identity"
simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                   link=link)
features, labels = simu.simulate()
# features = features[labels != 0, :]
# labels = labels[labels != 0]

# features = scipy.sparse.csr_matrix(features)
print('features', features)
model = ModelPoisReg(fit_intercept=False, link=link)
model.fit(features, labels)

l_l2sq = 1

prox = ProxL1(1e-1)
sdca = SDCA(l_l2sq, max_iter=200, print_every=10, tol=1e-16, batch_size=1)
sdca.set_model(model).set_prox(prox)
sdca.solve()

# newton = Newton(max_iter=10, print_every=1, tol=1e-16)
# newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
# newton.solve(weights)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))

plot_history([sdca], log_scale=True, dist_min=True, ax=axes[0])

ax = axes[1]
# print("sdca.get_iterate_history", sdca._solver.get_iterate_history()[-1])
print('get_primal_vector', sdca._solver.get_primal_vector())
iterate = sdca._solver.get_primal_vector()

print('GRAD',  prox.call(iterate - model.grad(iterate) - l_l2sq * iterate) - iterate)
print('zero labels proportion * l1= ', sum(labels == 0) / len(labels) * prox.strength)

n = sum(labels != 0)
n_0 = len(labels)
fake_prox = ProxL1(prox.strength * n / n_0)
print('strenghts', fake_prox.strength, prox.strength)
print('fake GRAD',  fake_prox.call(iterate - model.grad(iterate) - l_l2sq * iterate) - iterate)
# print("sdca.dual_solution", sdca.dual_solution)
# print(max(sdca.dual_solution))
ax.stem(sdca.solution, bins=30)
axes[2].stem(weights, bins=30)

plt.show()
