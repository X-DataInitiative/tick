import tick

import numpy as np
import matplotlib.pyplot as plt

from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems


def run_solvers(model, l_l2sq, ax_list):
    solvers = []
    coeff0 = np.ones(model.n_coeffs)

    # model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
    # model_dual.fit(features, labels)
    # max_iter_dual_bfgs = 1000
    # lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
    #                      print_every=int(max_iter_dual_bfgs / 7))
    # lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
    # lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
    # print(lbfgsb_dual.solution.mean())
    # print(model_dual.get_primal(lbfgsb_dual.solution))
    # for i, x in enumerate(lbfgsb_dual.history.values['x']):
    #      primal = lbfgsb._proj.call(model_dual.get_primal(x))
    #      lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)
    #

    max_iter_sdca = 1000
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-10)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()
    solvers += [sdca]

    sdca_2 = SDCA(l_l2sq, max_iter=max_iter_sdca,
                  print_every=int(max_iter_sdca / 7), tol=1e-10, batch_size=2)
    sdca_2.set_model(model).set_prox(ProxZero())
    sdca_2.solve()
    solvers += [sdca_2]

    lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=1e-10)
    lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    lbfgsb.solve(coeff0)
    solvers += [lbfgsb]

    svrg = SVRG(max_iter=100, print_every=10, tol=1e-10, step=1e-1)
    svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    svrg.solve(coeff0)
    solvers += [svrg]

    scpg = SCPG(max_iter=100, print_every=10, tol=1e-10, step=1e-3)
    scpg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    scpg.solve(coeff0)
    solvers += [scpg]

    # print(model.n_coeffs)
    newton = Newton(max_iter=100, print_every=10)
    newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    newton.solve(coeff0)
    solvers += [newton]

    plot_history(solvers, dist_min=True, log_scale=True,
                 x='time', ax=ax_list[0])

    ax_list[1].stem(newton.solution, linefmt='b-', markerfmt='bo', basefmt='b-')


dataset = 'crime'
features, labels = fetch_poisson_dataset(dataset)

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_2sq_list = [1e-2, 1e-3, 1e-4, 1. / np.sqrt(len(labels))]
fig, ax_list_list = plt.subplots(2, len(l_2sq_list))
for i, l_2sq in enumerate(l_2sq_list):
    run_solvers(model, l_2sq, ax_list_list[:, i])
    ax_list_list[0, i].set_title('$\\lambda = {:.3g}$'.format(l_2sq))

plt.show()
