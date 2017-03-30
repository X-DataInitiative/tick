import matplotlib.pyplot as plt
import numpy as np
import scipy
from pylab import rcParams

from tick.plot import plot_history
from tick.simulation import SimuLogReg
from tick.optim.solver import SGD, SVRG, AGD, AdaGrad
from tick.optim.model import ModelLogReg
from tick.optim.prox import ProxL2Sq

rcParams['figure.figsize'] = 16, 4

# We simulate logistic data with feature vector w and intercept c
n_features, n_samples = 10, 10000
w = np.random.normal(0, 1, n_features)
c = 0.2
sim = SimuLogReg(weights=w, intercept=c, n_samples=n_samples)
sim.simulate()
X = sim.features
y = sim.labels

# We create the corresponding model and a prox L2
model = ModelLogReg(fit_intercept=True).fit(X, labels=y)
l_l2sq = 1e-7
prox = ProxL2Sq(strength=l_l2sq)
x0 = np.zeros(n_features + 1)

# We get the reference solution thanks to scipy
# It will be used in order to examine how close our solvers can get to the
# correct answer.
func = lambda x: model.loss(x) + prox.value(x)
fprime = lambda x: model.grad(x) + prox.strength * x
minimizer = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=fprime)[0]
minimum = func(minimizer)

# SVRG solver
svrg = SVRG(max_iter=100, print_every=3, record_every=1, tol=1e-10)
svrg.set_model(model)
svrg.set_prox(prox)
svrg.history.set_minimizer(minimizer)
svrg.history.set_minimum(minimum)
step = 1. / model.get_lip_max()
svrg.solve(x0, step)

# AGD solver
agd = AGD(max_iter=100, print_every=3, record_every=1, tol=1e-10)
agd.set_model(model)
agd.set_prox(prox)
agd.history.set_minimizer(minimizer)
agd.history.set_minimum(minimum)
agd.solve(x0)

# SGD solver
sgd = SGD(max_iter=100, print_every=10, record_every=1, tol=1e-10, seed=1516,
          rand_type='perm')
sgd.set_model(model)
sgd.set_prox(prox)
sgd.history.set_minimizer(minimizer)
sgd.history.set_minimum(minimum)
sgd.solve(x0, 5)

# AdaGrad solver
adagrad = AdaGrad(max_iter=100, print_every=10, record_every=1, tol=1e-10,
                  seed=1516, rand_type='perm')
adagrad.set_model(model)
adagrad.set_prox(prox)
adagrad.history.set_minimizer(minimizer)
adagrad.history.set_minimum(minimum)
adagrad.solve(x0)


# We plot our solvers results
solvers = [svrg, sgd, agd, adagrad]

ax1 = plt.subplot(121)
plot_history(solvers, ax=ax1, y='dist_obj', log_scale=True)
ax1.set_ylabel("Relative distance to best objective")

ax2 = plt.subplot(122)
plot_history(solvers, ax=ax2, y='rel_obj', x='time', log_scale=True,)
ax2.set_ylabel("Objective change among iterations")

plt.show()
