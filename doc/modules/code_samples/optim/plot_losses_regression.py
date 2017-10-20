import matplotlib.pyplot as plt
import numpy as np

from tick.optim.model import ModelLinReg, ModelEpsilonInsensitive, \
    ModelHuber, ModelAbsoluteRegression

n = 1000
x = np.linspace(-2, 2, n)

ModelClasses = [ModelLinReg, ModelHuber, ModelEpsilonInsensitive,
                ModelAbsoluteRegression]

plt.figure(figsize=(8, 6))
for Model in ModelClasses:
    model = Model(fit_intercept=False) \
        .fit(np.array([[1.]]), np.array([0.]))
    y = [model.loss(np.array([t])) for t in x]
    plt.plot(x, y, lw=4, label=model.name)

plt.xlabel(r"$y'$", fontsize=16)
plt.ylabel(r"$y' \mapsto \ell(0, y')$", fontsize=16)
plt.title('Losses for regression', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
