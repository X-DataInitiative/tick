import matplotlib.pyplot as plt
import numpy as np

from tick.optim.model import ModelHinge, ModelModifiedHuber, \
    ModelQuadraticHinge, ModelSmoothedHinge, ModelLogReg

n = 1000
x = np.linspace(-3, 2, n)

ModelClasses = [ModelHinge, ModelModifiedHuber, ModelQuadraticHinge,
                ModelSmoothedHinge, ModelLogReg]

plt.figure(figsize=(8, 6))
for Model in ModelClasses:
    model = Model(fit_intercept=False) \
        .fit(np.array([[1.]]), np.array([1.]))
    y = [model.loss(np.array([t])) for t in x]
    plt.plot(x, y, lw=4, label=model.name)

plt.xlabel(r"$y y'$", fontsize=18)
plt.ylabel(r"$y y' \mapsto \ell(y, y')$", fontsize=18)
plt.title('Losses for binary classification', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=18)
plt.tight_layout()
