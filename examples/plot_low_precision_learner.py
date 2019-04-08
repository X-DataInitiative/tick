"""
===========================
Precision vs speed tradeoff
===========================

In this example we compare the convergence speed of our learners given the
float precision used.

In both case the convergence speed in term of number of iterations
(on the left) is similar up to float 32 precision.
But compared to the running time (on the right), we can see that using
float 32 instead of float 64 leads to faster convergence up to
float 32 precision.
"""
import matplotlib.pyplot as plt

from tick.dataset import fetch_tick_dataset
from tick.linear_model import LogisticRegression
from tick.plot import plot_history

X, y = fetch_tick_dataset('binary/adult/adult.trn.bz2')
X = X.toarray()  # It is more visible with dense matrices

max_iter = 50
seed = 7108

learner_64 = LogisticRegression(tol=0, max_iter=max_iter, record_every=2,
                                random_state=seed)
learner_64.fit(X, y)

X_32, y_32 = X.astype('float32'), y.astype('float32')
learner_32 = LogisticRegression(tol=0, max_iter=max_iter, record_every=2,
                                random_state=seed)
learner_32.fit(X_32, y_32)

# For a fair comparison, we access private attributes to compute both
# objective with float 64 precision
learner_32._solver_obj.history.values['obj'] = [
    learner_64._solver_obj.objective(coeffs.astype('float64'))
    for coeffs in learner_32._solver_obj.history.values['x']
]

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
plot_history([learner_32, learner_64], x='n_iter',
             labels=['float 32',
                     'float 64'], dist_min=True, log_scale=True, ax=axes[0])
plot_history([learner_32, learner_64], x='time',
             labels=['float 32',
                     'float 64'], dist_min=True, log_scale=True, ax=axes[1])

axes[0].set_ylabel(r'$\frac{f(w^t) - f(w^*)}{f(w^*)}$')
axes[0].set_xlabel('n epochs')
axes[1].set_ylabel('')
axes[1].set_xlabel('time (s)')

plt.show()
