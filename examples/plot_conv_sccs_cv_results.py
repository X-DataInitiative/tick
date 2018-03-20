"""
====================================================================
ConvSCCS cross validation on simulated longitudinal features example
====================================================================

In this example we simulate longitudinal data with preset relative incidence
for each feature. We then perform a cross validation of the ConvSCCS model
and compare the estimated coefficients to the relative incidences used for
the simulation.
"""
from time import time
import numpy as np
from scipy.sparse import csr_matrix, hstack
from matplotlib import cm
import matplotlib.pylab as plt
from tick.survival.simu_sccs import CustomEffects
from tick.survival import SimuSCCS, ConvSCCS
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Simulation parameters
seed = 0
lags = 49
n_samples = 2000
n_intervals = 750
n_corr = 3

# Relative incidence functions used for the simulation
ce = CustomEffects(lags + 1)
null_effect = [ce.constant_effect(1)] * 2
intermediate_effect = ce.bell_shaped_effect(2, 30, 15, 15)
late_effects = ce.increasing_effect(2, curvature_type=4)

sim_effects = [*null_effect, intermediate_effect, late_effects]

n_features = len(sim_effects)
n_lags = np.repeat(lags, n_features).astype('uint64')

coeffs = np.log(np.hstack(sim_effects))

# Time drift (age effect) used for the simulations.
time_drift = lambda t: np.log(8 * np.sin(.01 * t) + 9)

# Simaltion of the features.
sim = SimuSCCS(n_samples, n_intervals, n_features, n_lags,
               time_drift=time_drift, coeffs=coeffs, seed=seed,
               n_correlations=n_corr, verbose=False)
features, censored_features, labels, censoring, coeffs = sim.simulate()

# Plot the Hawkes kernel matrix used to generate the features.
fig, ax = plt.subplots(figsize=(7, 6))
heatmap = ax.pcolor(sim.hawkes_exp_kernels.adjacency, cmap=cm.Blues)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
fig.colorbar(heatmap, cax=cax)
ax.set_title('Hawkes adjacency matrix used for the simulation');
plt.show()

## Add age_groups features to feature matrices.
agegrps = [0, 125, 250, 375, 500, 625, 750]
n_agegrps = len(agegrps) - 1

feat_agegrp = np.zeros((n_intervals, n_agegrps))
for i in range(n_agegrps):
    feat_agegrp[agegrps[i]:agegrps[i + 1], i] = 1

feat_agegrp = csr_matrix(feat_agegrp)
features = [hstack([f, feat_agegrp]).tocsr() for f in features]
censored_features = [hstack([f, feat_agegrp]).tocsr() for f in
                     censored_features]
n_lags = np.hstack([n_lags, np.zeros(n_agegrps)])

# Learning
# Example code for cross validation
# start = time()
# learner = ConvSCCS(n_lags=n_lags.astype('uint64'),
#                    penalized_features=np.arange(n_features),
#                    random_state=42)
# strength_TV_range = (-4, -1)
# strength_L1_range = (-5, -2)
# _, cv_track = learner.fit_kfold_cv(features, labels, censoring,
#                            strength_TV_range, strength_L1_range,
#                            bootstrap=True, bootstrap_rep=50, n_cv_iter=50)
# elapsed_time = time() - start
# print("Elapsed time (model training): %.2f seconds \n" % elapsed_time)
# print("Best model hyper parameters: \n", cv_track.best_model['strength'])
# cv_track.plot_cv_report(35, 45)
# plt.show()

# using the parameters resulting from cross-validation
learner = ConvSCCS(n_lags=n_lags.astype('uint64'),
                   penalized_features=np.arange(n_features),
                   random_state=42, strength_tv=0.0036999724314638062,
                   strength_group_l1=0.0001917004158917066)

_, bootstrap_ci = learner.fit(features, labels, censoring,
                              bootstrap=True, bootstrap_rep=20)

# Plot estimated parameters
# when using cross validation output:
# bootstrap_ci = cv_track.best_model['bootstrap_ci']
# get bootstrap confidence intervals
refitted_coeffs = bootstrap_ci['refit_coeffs']
lower_bound = bootstrap_ci['lower_bound']
upper_bound = bootstrap_ci['upper_bound']

n_plots = np.sum([1 for l in n_lags if l > 0])
n_rows = int(np.ceil(n_plots / 2))
fig, axarr = plt.subplots(n_rows, 2, sharex=True, sharey=True, figsize=(10, 6))
offset = 0
c = 0
for l in n_lags:
    if l == 0:
        offset += 1
        continue
    end = int(offset + l + 1)
    m = refitted_coeffs[offset:end]
    lb = lower_bound[offset:end]
    ub = upper_bound[offset:end]
    ax = axarr[c // 2][c % 2]
    ax.plot(np.exp(coeffs[offset:end]), label="True RI")
    ax.step(np.arange(l+1), np.exp(m), label="Estimated RI")
    ax.fill_between(np.arange(l + 1), np.exp(lb), np.exp(ub), alpha=.5,
                    color='orange', step='pre', label="95% boostrap CI")
    offset = end
    c += 1

plt.suptitle('Estimated relative risks with 95% confidence bands')
axarr[0][1].legend(loc='upper right')
axarr[0][0].set_ylabel('Relative incidence')
axarr[1][0].set_ylabel('Relative incidence')
axarr[-1][0].set_xlabel('Time after exposure start')
axarr[-1][1].set_xlabel('Time after exposure start')
plt.show()

normalize = lambda x: x / np.sum(x)
offset = n_features * (lags + 1)
m = np.repeat(refitted_coeffs[offset:], 125)
lb = np.repeat(lower_bound[offset:], 125)
ub = np.repeat(upper_bound[offset:], 125)
plt.figure()
plt.plot(np.arange(n_intervals),
         normalize(np.exp(time_drift(np.arange(n_intervals)))))
plt.step(np.arange(n_intervals), normalize(np.exp(m)))
plt.fill_between(np.arange(n_intervals), np.exp(lb) / np.exp(m).sum(),
                 np.exp(ub) / np.exp(m).sum(), alpha=.5, color='orange',
                 step='pre')
plt.xlabel('Age')
plt.ylabel('Normalized Age Relative Incidence')
plt.title("Normalized age effect with 95% confidence bands");
plt.show()
