# L1

from experiments.hawkes_coeffs import dim_from_n
from tick.prox import ProxL1, ProxL1w, ProxNuclear
from tick.hawkes import ModelHawkesExpKernLeastSq


def get_n_decays_from_model(model):
    if isinstance(model, ModelHawkesExpKernLeastSq):
        return 1
    else:
        return model.n_decays


def create_prox_l1_no_mu(strength, model):
    dim = dim_from_n(model.n_coeffs, get_n_decays_from_model(model))
    prox = ProxL1(strength, positive=True, range=(dim, dim * dim + dim))
    return prox


def create_prox_l1w_no_mu(strength, model):
    weights = model.compute_penalization_constant(strength=strength)
    n_decays = get_n_decays_from_model(model)
    dim = dim_from_n(model.n_coeffs, n_decays)
    weights = weights[dim:]
    prox = ProxL1w(1, positive=True, weights=weights,
                   range=(dim, dim * dim * n_decays + dim))
    return prox


def create_prox_l1w_no_mu_un(strength, model):
    weights = model.compute_penalization_constant()
    n_decays = get_n_decays_from_model(model)
    dim = dim_from_n(model.n_coeffs, n_decays)
    weights = weights[dim:]
    prox = ProxL1w(strength, positive=True, weights=weights,
                   range=(dim, dim * dim * n_decays + dim))
    return prox

# Nuclear
def create_prox_nuclear(strength, model):
    dim = dim_from_n(model.n_coeffs, get_n_decays_from_model(model))
    prox_range = (dim, dim * dim + dim)
    n_rows = dim
    prox = ProxNuclear(strength, n_rows, range=prox_range, positive=True)
    return prox

def create_prox_l1w_no_mu_nuclear(strength, model):
    l1, tau = strength
    prox_l1 = create_prox_l1w_no_mu(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model)
    return (prox_l1, prox_nuclear,)

def create_prox_l1w_un_no_mu_nuclear(strength, model):
    l1, tau = strength
    prox_l1 = create_prox_l1w_no_mu_un(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model)
    return (prox_l1, prox_nuclear,)


def create_prox_l1_no_mu_nuclear(strength, model):
    l1, tau = strength
    prox_l1 = create_prox_l1_no_mu(l1, model)
    prox_nuclear = create_prox_nuclear(tau, model)
    return prox_l1, prox_nuclear

