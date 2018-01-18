"""
==============================
Cox regression data simulation
==============================

Generates Cox Regression realization given a weight vector 
"""

import matplotlib.pyplot as plt
import numpy as np
from tick.survival import SimuCoxReg

n_samples = 150
weights = np.array([0.3, 1.2])

seed = 123
simu_coxreg = SimuCoxReg(weights, n_samples=n_samples, seed=123, verbose=False)
X, T, C = simu_coxreg.simulate()

plt.figure(figsize=(6, 4))

plt.scatter(*X[C == 0].T, c=T[C == 0], cmap='RdBu', marker="x",
            label="censoring")
plt.scatter(*X[C == 1].T, c=T[C == 1], cmap='RdBu', marker="o",
            label="failure")
plt.colorbar()
plt.legend(loc='upper left')
plt.title('Cox regression', fontsize=16)
