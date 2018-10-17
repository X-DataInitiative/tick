import numpy as np
import matplotlib.pyplot as plt
import scipy

from tick.plot import stems, plot_history

from tick.prox import ProxZero, ProxL2Sq, ProxL1

from tick.linear_model import ModelPoisReg, SimuPoisReg, SimuLogReg, \
    ModelLogReg
from tick.solver import SDCA
from tick.solver.newton import Newton
from tick.solver.sdca import AtomicSDCA
from tick.plot.plot_utilities import share_x, share_y

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

seed = 2139

link = 'identity'

def simulate_sparse_matrix(n_samples, n_features, sparsity):
    class CustomRandomState:

        def __init__(self, n_rows, n_cols, p=None):
            self.n_rows = n_rows
            self.n_cols = n_cols
            if p is None:
                p = 10 / n_cols
            self.p = p

        def rand(self, size):
            return np.random.rand(size)

        def randint(self, k):
            col = min(np.random.geometric(self.p), self.n_cols) - 1
            row = np.random.randint(self.n_rows)
            return col * self.n_rows + row

    rs = CustomRandomState(n_samples, n_features, p=None)
    S = scipy.sparse.random(n_samples, n_features, density=sparsity,
                            random_state=rs, format='csr')

    non_zeros_S = S != 0
    int_S = S
    int_S[int_S != 0] = 1

    # print('repartition accross columns')
    # print(non_zeros_S.mean(axis=0))

    n_random = 10
    means, means_n = [], []
    for i in range(n_random):
        means += [non_zeros_S.dot(non_zeros_S[i].toarray().reshape(-1)).mean()]
        means_n += [int_S.dot(int_S[i].toarray().reshape(-1)).mean()]
    print('Intersections probablity {}% +- {:.2f}%'
          .format(np.mean(means) * 100,
                  196 * np.std(means) / np.sqrt(len(means))))
    print('Average number of intersections {}% +- {:.2f}%'
          .format(np.mean(means_n),
                  1.96 * np.std(means_n) / np.sqrt(len(means_n))))

    return S

def simulate_poisson(n_samples, n_features, sparsity=1e-3):
    np.random.seed(23982)

    nnz = int(0.3 * n_features)
    weights = np.random.normal(size=n_features)
    mask_weights = np.random.choice(np.arange(n_features), nnz, replace=False)
    weights[mask_weights] = 0
    # weights = np.abs(weights)
    weights /= (n_features * sparsity)

    # weights /= nnz
    # print(np.linalg.norm(weights) ** 2 / n_features)

    np.random.seed(2309)

    epsilon = 1e-6
    features = simulate_sparse_matrix(n_samples, n_features, sparsity)

    features = np.abs(features)
    features = features[features.dot(weights) > epsilon, :]

    while features.shape[0] < n_samples:
        new_features = simulate_sparse_matrix(n_samples, n_features, sparsity)
        new_features = np.abs(new_features)
        new_features = new_features[new_features.dot(weights) > epsilon, :]

        features = scipy.sparse.vstack([features, new_features])

    features = features[:n_samples, :]
    # features /= (n_features * sparsity)
    magnitudes = np.random.rand(n_samples) * 4
    diag = scipy.sparse.csr_matrix((n_samples, n_samples))
    diag.setdiag(magnitudes)
    features = diag * features

    print('prod mean 2', features.dot(weights).mean())
    print('features', features.shape)

    simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                       link=link)
    features, labels = simu.simulate()

    print('% non zeros = ', np.mean(labels != 0))
    return features, labels, weights


def simulate_poisson2(n_samples, n_features=None, sparsity=None):
    if n_features is None:
        n_features = 100

    nnz = int(0.3 * n_features)

    np.random.seed(239829)

    weights = np.random.normal(size=n_features)
    mask_weights = np.random.choice(np.arange(n_features), nnz, replace=False)
    weights[mask_weights] = 0
    # weights = np.abs(weights)

    weights /= nnz
    # print(np.linalg.norm(weights) ** 2 / n_features)

    np.random.seed(2309)

    features = np.random.randn(n_samples, n_features)
    features = np.abs(features)
    features /= n_features
    print('sum', np.sum(features.dot(weights) > 0))
    print('prod mean 1', features.dot(weights).mean())
    print('prod mean 1.4',
          (features.dot(weights)[features.dot(weights) < 0]).mean())
    print('prod mean 1.4', (features.dot(weights)[features.dot(weights) > 0]).mean())

    epsilon = 1e-1
    while features.dot(weights).min() <= epsilon:
        n_fail = sum(features.dot(weights) <= epsilon)
        features[features.dot(weights) <= epsilon] = \
            np.random.randn(n_fail, n_features)
        features = np.abs(features)

    simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                       link='identity')
    print('prod mean 2', features.dot(weights).mean())
    features, labels = simu.simulate()
    print('% non zeros = ', np.mean(labels != 0))
    return features, labels, weights


def compute_l_l2sq(features, labels):
    non_zero_features = features[labels != 0]
    n = non_zero_features.shape[0]
    if scipy.sparse.issparse(features):
        norms = scipy.sparse.linalg.norm(non_zero_features, axis=1)
    else:
        norms = np.linalg.norm(non_zero_features, axis=1)
    mean_features_norm = np.mean(norms) ** 2

    return mean_features_norm / n


dense = False
if dense:
    features, labels, weights = simulate_poisson2(
        n_samples=5000, n_features=1000)
else:
    features, labels, weights = simulate_poisson(
        n_samples=12000, n_features=5000, sparsity=0.1)
        # n_samples=3000, n_features=1000, sparsity=1e-2)

l_l2sq = compute_l_l2sq(features, labels) * 1e-2
print('l_l2sq', l_l2sq)
model = ModelPoisReg(fit_intercept=False, link=link)
model.fit(features, labels)

prox = ProxZero()  # ProxL1(1e-5)

batch_sizes = [1, 2, 3, 10]
test_n_threads = [1, 2]# , 3, 4]#, 3, 4, 8]

fig, axes = plt.subplots(2, len(batch_sizes) + 1, figsize=(3 * len(batch_sizes) + 3, 6))
axes = axes.reshape(2, -1)


max_iter = 1000
for i, batch_size in enumerate(batch_sizes):
    axes[0, i].set_title('batch {}'.format(batch_size))
    solver_list = []
    solver_labels = []

    for n_threads in test_n_threads:

        args = [l_l2sq]
        kwargs = dict(tol=1e-14, verbose=False, seed=seed, max_iter=max_iter,
                      batch_size=batch_size, record_every=int(max_iter / 20))
        if n_threads == 1:
            solver = SDCA(*args, **kwargs)
            solver.set_model(model).set_prox(prox)
            solver_labels += [solver.name]
        else:
            solver = AtomicSDCA(*args, **kwargs, n_threads=n_threads)
            solver.set_model(model).set_prox(prox)
            solver_labels += ['{} {}'.format(solver.name, n_threads)]

        print('-'*20 + '\n', solver.name)
        solver.solve()
        solver_list += [solver]

        if n_threads == 1:
            print('DUALITY GAP',
        solver.objective(solver.solution) - solver.dual_objective(solver.dual_solution)
        )
            iterate = solver.solution
            print('GRAD', np.abs(prox.call(
                iterate - model.grad(iterate) - l_l2sq * iterate) - iterate).mean())
            print('OBJECTIVE', solver.objective(iterate))
            print('LOSS', model.loss(iterate))

    # solver.print_history()

    plot_history(solver_list, y="dual_objective", x="time", dist_min=True, log_scale=True,
                 labels=solver_labels, ax=axes[0, i])

    plot_speedup = False

    if not plot_speedup:
        plot_history(solver_list, y='dual_objective', x="n_iter", dist_min=True, log_scale=True,
                     labels=solver_labels, ax=axes[1, i])
    else:

        solver_objectives = np.array([
            solver.history.values['obj'] for solver in solver_list])

        dist_solver_objectives = solver_objectives - np.nanmin(solver_objectives)

        # speedup
        ax = axes[1, i]
        ax.plot([test_n_threads[0], test_n_threads[-1]], [1, test_n_threads[-1]],
                ls='--', lw=1, c='black')

        for target_precision in [1e-4, 1e-6, 1e-8, 1e-10]:
            target_indexes = [
                np.argwhere(dist_solver_objectives[i] < target_precision)[0][0]
                if np.nanmin(dist_solver_objectives[i]) < target_precision
                else np.nan
                for i in range(len(dist_solver_objectives))]
            print(target_precision, target_indexes)

            target_times = np.array([
                solver.history.values['time'][index]
                if not np.isnan(index)
                else np.nan
                for index, solver in zip(target_indexes, solver_list)])

            time_one = target_times[0]
            y = time_one / target_times
            ax.plot(test_n_threads, y, marker='x', label='{:.1g}'
                    .format(target_precision))
            ax.set_xlabel('number of cores')
            ax.set_ylabel('speedup')
            ax.set_title(solver_list[0].name)

            ax.legend()

bins = 30
axes[0, -1].hist(solver_list[0].solution, alpha=0.5, bins=bins, label='$w_*$')
axes[0, -1].hist(weights, alpha=0.5, bins=bins, label='$w_0$')
axes[0, -1].set_yscale("log")
axes[0, -1].legend()
axes[1, -1].hist(solver_list[0].dual_solution)
axes[1, -1].set_yscale("log")

share_x(axes[0, :-1].reshape(1, -1))
share_x(axes[1, :-1].reshape(1, -1))
share_y(axes[0, :-1].reshape(1, -1))
share_y(axes[1, :-1].reshape(1, -1))
fig.tight_layout()
plt.show()
