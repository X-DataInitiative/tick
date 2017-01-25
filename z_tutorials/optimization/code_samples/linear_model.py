import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from mlpp.optim.model import ModelLinReg, ModelLogReg, ModelPoisReg
from mlpp.optim.solver import SDCA, SVRG, BFGS, Ista, Fista
from mlpp.optim.prox import ProxZero, ProxL2Sq
from mlpp.simulation import SimuLinReg, SimuLogReg, SimuPoisReg

seed = 1398
np.random.seed(seed)


def create_model(model_type, n_samples, n_features, with_intercept=True):
    weights = np.random.randn(n_features)
    intercept = None
    if with_intercept:
        intercept = np.random.normal()

    if model_type == 'Poisson':
        # we need to rescale features to avoid overflows
        weights /= n_features
        if intercept is not None:
            intercept /= n_features

    if model_type == 'Linear':
        simulator = SimuLinReg(weights, intercept=intercept, n_samples=n_samples,
                               verbose=False)
    elif model_type == 'Logistic':
        simulator = SimuLogReg(weights, intercept=intercept, n_samples=n_samples,
                               verbose=False)
    elif model_type == 'Poisson':
        simulator = SimuPoisReg(weights, intercept=intercept, n_samples=n_samples,
                                verbose=False)

    labels, features = simulator.simulate()

    if model_type == 'Linear':
        model = ModelLinReg(fit_intercept=with_intercept)
    elif model_type == 'Logistic':
        model = ModelLogReg(fit_intercept=with_intercept)
    elif model_type == 'Poisson':
        model = ModelPoisReg(fit_intercept=with_intercept)

    model.fit(labels, features)
    return model


def run_solvers(model, l_l2sq):
    try:
        svrg_step = 1. / model.get_lip_max()
    except AttributeError:
        svrg_step = 1e-3
    try:
        ista_step = 1. / model.get_lip_best()
    except AttributeError:
        ista_step = 1e-1

    bfgs = BFGS(verbose=False, tol=1e-13)
    bfgs.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    bfgs.solve()
    bfgs.history.set_minimizer(bfgs.solution)
    bfgs.history.set_minimum(bfgs.objective(bfgs.solution))
    bfgs.solve()

    svrg = SVRG(step=svrg_step, verbose=False, tol=1e-10, seed=seed)
    svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    svrg.history.set_minimizer(bfgs.solution)
    svrg.history.set_minimum(bfgs.objective(bfgs.solution))
    svrg.solve()

    sdca = SDCA(l_l2sq, verbose=False, seed=seed, tol=1e-10)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.history.set_minimizer(bfgs.solution)
    sdca.history.set_minimum(bfgs.objective(bfgs.solution))
    sdca.solve()

    ista = Ista(verbose=False, tol=1e-10, step=ista_step, linesearch=False)
    ista.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    ista.history.set_minimizer(bfgs.solution)
    ista.history.set_minimum(bfgs.objective(bfgs.solution))
    ista.solve()

    fista = Fista(verbose=False, tol=1e-10, step=ista_step, linesearch=False)
    fista.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    fista.history.set_minimizer(bfgs.solution)
    fista.history.set_minimum(bfgs.objective(bfgs.solution))
    fista.solve()

    return bfgs, svrg, sdca, ista, fista


model_types = ['Linear', 'Logistic', 'Poisson']
l_l2sqs = [1e-3, 1e-2, 1e-1]

fig, axes = plt.subplots(len(model_types), len(l_l2sqs),
                         figsize=(4 * len(l_l2sqs), 3 * len(model_types)),
                         sharey=True, sharex=True)

n_samples = 1000
n_features = 20

for (model_type, l_l2sq), ax in zip(product(model_types, l_l2sqs),
                                    axes.ravel()):
    model = create_model(model_type, n_samples, n_features)

    bfgs, svrg, sdca, ista, fista = run_solvers(model, l_l2sq)
    bfgs.history.plot(y_axis='dist_obj', ax=ax, labels=['BFGS'])
    svrg.history.plot(y_axis='dist_obj', ax=ax, labels=['SVRG'])
    sdca.history.plot(y_axis='dist_obj', ax=ax, labels=['SDCA'])
    ista.history.plot(y_axis='dist_obj', ax=ax, labels=['ISTA'])
    fista.history.plot(y_axis='dist_obj', ax=ax, labels=['FISTA'])
    ax.semilogy()
    ax.legend_.remove()
    ax.set_xlabel('')
    ax.set_ylim([1e-9, 1])

for l_l2sq, ax in zip(l_l2sqs, axes[0]):
    ax.set_title('$\lambda = %.2g$' % l_l2sq)

for model_type, ax in zip(model_types, axes):
    ax[0].set_ylabel('%s regression' % model_type, fontsize=17)

for ax in axes[-1]:
    ax.set_xlabel('epochs')

axes[-1][1].legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=5)
plt.show()
