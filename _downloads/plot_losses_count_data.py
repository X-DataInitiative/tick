import matplotlib.pyplot as plt
import numpy as np

from tick.optim.model import ModelPoisReg

n = 1000
x = np.linspace(-1.5, 2, n)

models = [
    ModelPoisReg(fit_intercept=False, link='exponential'),
    ModelPoisReg(fit_intercept=False, link='identity')
]

labels = [
    "ModelPoisReg(link='exponential')",
    "ModelPoisReg(link='identity')"
]
plt.figure(figsize=(8, 6))
for model, label in zip(models, labels):
    model.fit(np.array([[1.]]), np.array([1.]))
    y = [model.loss(np.array([t])) for t in x]
    plt.plot(x, y, lw=4, label=label)

plt.xlabel(r"$y'$", fontsize=16)
plt.ylabel(r"$y' \mapsto \ell(1, y')$", fontsize=16)
plt.title('Losses for count data', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
