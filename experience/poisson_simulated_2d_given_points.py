import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.optimize import fmin_l_bfgs_b

from experience.poisreg_sdca import ModelPoisRegSDCA
from tick.optim.prox import ProxL2Sq, ProxZero
from tick.optim.model import ModelPoisReg
from tick.optim.solver import LBFGSB, Newton, SDCA

seaborn.set_style('white')

COLORS = {0: 'C0', 1: 'C6', 2: 'C2'}


def objective(x, y):
    best_objective = newton.history.last_values['obj']
    return np.log(newton.objective(np.array([x, y])) - best_objective)


def dual_objective(u, v):
    best_dual_objective = sdca.dual_objective(sdca.dual_solution)
    return np.log(best_dual_objective - sdca.dual_objective(np.array([u, v])))


def dual_grad_vector(u, v):
    dual_vector = np.array([u, v])
    # n_samples = len(labels)
    # non_zero_features = features[labels != 0]
    #
    # alpha_x = np.sum(np.diag(dual_vector).dot(non_zero_features), axis=0)
    # psi_x = np.sum(features, axis=0).dot(non_zero_features.T)
    #
    # grad = 1. / n_samples * labels[labels != 0] / dual_vector
    # grad -= 1. / (l_l2sq * n_samples**2) * alpha_x.dot(non_zero_features.T)
    # grad += 1. / (l_l2sq * n_samples**2) * psi_x
    # return - grad
    return model_sdca.grad(dual_vector)


def grad_vector(x, y):
    coeff = np.array([x, y])
    return newton.model.grad(coeff) + l_l2sq * coeff


def log_grad_norm(x, y):
    return np.log(np.linalg.norm(grad_vector(x, y)))


def plot_obj(limits, ax, nx=100, ny=100, fun=log_grad_norm):
    x_min, x_max, y_min, y_max = limits

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_max, y_min, ny)

    plot_arrows = fun in [grad_vector, dual_grad_vector]
    if plot_arrows:
        U, V = np.empty((len(x), len(y))), np.empty((len(x), len(y)))
    else:
        Z = np.empty((len(x), len(y)))

    for i in range(x.size):
        for j in range(y.size):
            if plot_arrows:
                grad = fun(x[i], y[j])
                U[j, i] = - grad[0]
                V[j, i] = - grad[1]
            else:
                Z[j, i] = fun(x[i], y[j])

    if plot_arrows:
        max_tolerance = 3 * np.median(np.abs(U))
        mask_too_big = np.abs(U) > max_tolerance
        U[mask_too_big] = np.nan  # np.sign(U[mask_too_big]) * max_tolerance

        max_tolerance = 3 * np.median(np.abs(V))
        mask_too_big = np.abs(V) > max_tolerance
        V[mask_too_big] = np.nan  # np.sign(V[mask_too_big]) * max_tolerance

        ax.quiver(x, y, U, V, units='width', headwidth=3)

    else:
        im = ax.imshow(
            Z,
            extent=(x[0], x[-1], y[0], y[-1]),
            origin='lower')


def plot_scatter(limits, labels, features, ax):
    x_min, x_max, y_min, y_max = limits
    for label in np.unique(labels):
        mask = labels == label
        color = COLORS[label]
        ax.scatter(features[mask, 0], features[mask, 1], c=color,
                   label=r'$y_i = {}$'.format(label))

        for feature in features[mask]:
            ax.arrow(0, 0, feature[0], feature[1], color=color)

    ax.legend(fontsize=8)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])


def plot_feasible_set(limits, labels, features, ax):
    x_min, x_max, y_min, y_max = limits

    for feature in features[labels > 0]:
        ortho_0 = x_min
        ortho_1 = - ortho_0 * feature[0] / feature[1]
        ax.plot([0, ortho_0], [0, ortho_1], color='black')


def plot_steps(steps, ax, color=None, label=''):
    for i in range(1, len(steps)):
        before = steps[i - 1]
        after = steps[i]
        if i == 1:
            plot_label = label
        else:
            plot_label = None
        ax.plot([before[0], after[0]], [before[1], after[1]],
                color=color, label=plot_label)
        ax.scatter(after[0], after[1], color=color, s=10)


def plot_solver_steps(solver_list, ax):
    for i, solver in enumerate(solver_list):
        rel_deltas = np.array(solver.history.values['rel_delta'])
        max_rel_delta = np.maximum.accumulate(rel_deltas[::-1])
        last_significant = len(max_rel_delta) - \
                           np.searchsorted(max_rel_delta, 1e-2)
        solver_steps = solver.history.values['x'][:last_significant]
        plot_steps(solver_steps, ax, color=COLORS[i],
                   label='L-BFGS-B {}'.format(i + 1))


labels = np.array([0., 1., 2.])
features = np.array([
    [0.23, 1.1], [-0.17, -0.44], [-0.32, 0.80]
])

features *= 3

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_l2sq = 0.5
model_sdca = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
model_sdca.fit(features, labels)


newton = Newton()
newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(0.2 * np.ones(model.n_coeffs))

sdca = SDCA(l_l2sq)
sdca.set_model(model).set_prox(ProxZero())
sdca.solve()

lbfgsb_list = []
start_points = [[-2.70, 0.03], [-0.62, -0.15], [-1.61, 0.47]]
for start_point in start_points:
    start_point = np.array(start_point)
    lbfgsb = LBFGSB(print_every=1, tol=1e-7, verbose=False)
    lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    lbfgsb.solve(start_point)
    lbfgsb_list += [lbfgsb]


def insp(x, this_dual_steps=None):
    this_dual_steps += [x.copy()]


dual_start_points = [[0.01, 1.], [3., 1e-5], [0.1, 0.1]]
dual_steps = []

for start_point in dual_start_points:
    start_point = np.array(start_point)
    # bounds = [(1e-10, None) for _ in range(sum(labels != 0))]

    dual_steps.append([])
    this_dual_steps = dual_steps[-1]
    this_dual_steps += [start_point]

    print('----')
    res = \
        fmin_l_bfgs_b(model_sdca.loss,
                      start_point,
                      model_sdca.grad,
                      maxiter=100, pgtol=1e-7,
                      callback=lambda x: insp(x,
                                              this_dual_steps=this_dual_steps),
                      # bounds=bounds,
                      disp=False)

fig, ax_list_list = plt.subplots(2, 3, figsize=(10.5, 7))
cst = 4
limits = (-cst, cst, cst, -cst)

resolution = 150

ax_list = ax_list_list[0]
plot_obj(limits, ax_list[0], nx=resolution, ny=resolution, fun=objective)
plot_scatter(limits, labels, features, ax_list[0])
plot_feasible_set(limits, labels, features, ax_list[0])

plot_obj(limits, ax_list[1], nx=resolution, ny=resolution, fun=log_grad_norm)
plot_obj(limits, ax_list[1], nx=30, ny=30, fun=grad_vector)

plot_obj(limits, ax_list[2], nx=resolution, ny=resolution, fun=log_grad_norm)
plot_solver_steps(lbfgsb_list, ax_list[2])

ax_list[2].legend(fontsize=8)

ax_list[0].set_title('Original datapoints and log-distance to\n'
                     'optimal objective on the feasible set',
                     fontsize=10)

ax_list[2].set_title('Paths taken by three L-BFGS-B solvers\n'
                     'starting from the feasible set',
                     fontsize=10)

ax_list = ax_list_list[1]
cst = 4
limits = (-1, cst, cst, -1)

plot_obj(limits, ax_list[0], nx=resolution, ny=resolution, fun=dual_objective)

plot_obj(limits, ax_list[1], nx=resolution, ny=resolution, fun=dual_objective)
plot_obj(limits, ax_list[1], nx=30, ny=30, fun=dual_grad_vector)

plot_obj(limits, ax_list[2], nx=resolution, ny=resolution, fun=dual_objective)
for i, this_dual_steps in enumerate(dual_steps):
    plot_steps(this_dual_steps, ax_list[2], color=COLORS[i],
               label='L-BFGS-B {}'.format(i + 1))

ax_list[0].set_title('log-distance to optimal dual objective\n'
                     'on the feasible set',
                     fontsize=10)

ax_list[1].set_title('Gradient field and log-distance to\n'
                     'optimal dual objective',
                     fontsize=10)

ax_list[2].set_title('Paths taken by three L-BFGS-B solvers\n'
                     'in the dual',
                     fontsize=10)

fig.tight_layout()
plt.show()
# plt.savefig('poisson_3_points_example.pdf')
