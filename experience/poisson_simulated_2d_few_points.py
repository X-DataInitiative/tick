import numpy as np
import matplotlib.pyplot as plt
import scipy

from tick.plot import stems, plot_history

from tick.optim.prox import ProxZero, ProxL2Sq

from tick.simulation import SimuPoisReg
from tick.optim.model import ModelPoisReg
from tick.optim.solver import LBFGSB, SDCA, Newton, SVRG

n_samples = 4
n_features = 2
nn_z = 1.

np.random.seed(320965)
# np.random.seed(3201932)

positive_weights = False
positive_features = False

weights = np.array([-1.0222822, 0.30856363]) * 2

features = np.random.randn(n_samples, n_features)

epsilon = 1e-1
while features.dot(weights).min() <= epsilon:
    n_fail = sum(features.dot(weights) <= epsilon)
    features[features.dot(weights) <= epsilon] = \
        np.random.randn(n_fail, n_features)

simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                   link='identity')
features, labels = simu.simulate()

print(features)
print(labels)

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_l2sq = 0.5
#
sdca = SDCA(l_l2sq, tol=1e-7)
sdca.set_model(model).set_prox(ProxZero())
sdca.solve()

lbfgsb_list = []

start_points = [[0.1, 0.1], [-0, -0.5], [-0.9, 0.9], [0, 0.1], [-0.53, -0.28]]
for start_point in start_points:
    start_point = np.array(start_point)
    lbfgsb = LBFGSB(print_every=1, tol=1e-7)
    lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    lbfgsb.solve(start_point)
    lbfgsb_list += [lbfgsb]

svrg_list = []
for start_point in start_points:
    start_point = np.array(start_point)
    svrg = SVRG(print_every=10, step=1e-4, tol=1e-7, seed=1029)
    svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    svrg.solve(start_point)
    svrg_list += [svrg]


def manual_grad(coeff):
    grad = l_l2sq * coeff
    for i in range(n_samples):
        grad += 1. / n_samples * features[i]
        if labels[i] != 0:
            grad -= 1 / n_samples * labels[i] / \
                    (coeff.dot(features[i])) * features[i]
    return grad


# print(features[labels != 0].dot(svrg.solution))

newton = Newton()
newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(0.2 * np.ones(model.n_coeffs))

print(sdca.solution)
print(lbfgsb.solution)
print(svrg.solution)
print(newton.solution)

# plot_history([sdca, newton], log_scale=True, dist_min=True, x='time')


plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='cool')
plt.colorbar()

#
# # plt.plot([0, newton.solution[0]], [0, newton.solution[1]])
# newton_arrow = plt.arrow(0, 0, newton.solution[0], newton.solution[1],
#                          head_width=0.05, head_length=0.1, fc='C1', ec='C1')
#
# svrg_arrow = plt.arrow(0, 0, svrg.solution[0], svrg.solution[1],
#                        head_width=0.05, head_length=0.1, fc='C2', ec='C2')
#
# original_arrow = plt.arrow(0, 0, weights[0], weights[1], head_width=0.05,
#                            head_length=0.1, fc='C3', ec='C3')
#
# plt.legend([newton_arrow, svrg_arrow, original_arrow],
#            ['newton & sdca', 'svrg', 'original'])
#
# plt.xlim([- 1.5 * np.abs(features[:, 0].max()), 1.5 * np.abs(features[:, 0].max())])
# plt.ylim([- 1.5 * np.abs(features[:, 1].max()), 1.5 * np.abs(features[:, 1].max())])
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def fun(x, y):
    # return x + 3 * y
    # return svrg.objective(np.array([x, y]))
    coeff = np.array([x, y])
    grad = svrg.model.grad(coeff) + l_l2sq * coeff
    return np.log(np.linalg.norm(grad))


def plot_obj(x_min, x_max, y_min, y_max, ax, nx=100, ny=100):
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_max, y_min, ny)
    Z = np.empty((len(x), len(y)))
    for i in range(x.size):
        for j in range(y.size):
            Z[j, i] = fun(x[i], y[j])

    Z[np.isinf(Z)] = np.nanmax(Z[np.isfinite(Z)])
    Z[np.isnan(Z)] = np.nanmax(Z)

    # s = fig.add_subplot(1, 1, 1, xlabel='$x$', ylabel='$y$')
    im = ax.imshow(
        Z,
        extent=(x[0], x[-1], y[0], y[-1]),
        origin='lower')
    # plt.colorbar(im)

    ax.arrow(0, 0, newton.solution[0], newton.solution[1],
             head_width=0.05, head_length=0.1, fc='C1', ec='C1')

#
fig, ax_list = plt.subplots(1, 2)
plot_obj(-3, 3, 3, -3, ax_list[0], nx=150, ny=150)
plot_obj(-10, 10, 10, -10, ax_list[1], nx=50, ny=50)
# plt.show()
# # plot_obj(0, 110, 0, -30, ax_list[1])
# plot_obj(-30, 110, 50, -100, ax_list[1], nx=150, ny=150)
#

svrg_steps = svrg.history.values['x']
#
#
def plot_steps(steps):
    for i in range(1, len(steps)):
        before = steps[i - 1]
        after = steps[i]
        ax_list[0].arrow(before[0], before[1], after[0] - before[0],
                         after[1] - before[1], head_width=0.05, head_length=0.1)
        ax_list[1].arrow(before[0], before[1], after[0] - before[0], after[1] - before[1])

for solver in lbfgsb_list:
    solver_steps = solver.history.values['x']
    plot_steps(solver_steps)

# print('SVRG steps')
# plot_steps(svrg_steps)
#
# print(model.features[labels != 0].dot(svrg.solution).max())
#
plt.show()
