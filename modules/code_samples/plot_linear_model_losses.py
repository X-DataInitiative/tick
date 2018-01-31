import matplotlib.pyplot as plt
import numpy as np

from tick.linear_model import ModelLinReg, ModelHinge, ModelQuadraticHinge, \
    ModelSmoothedHinge, ModelLogReg, ModelPoisReg
from tick.robust import ModelEpsilonInsensitive, ModelHuber, \
    ModelAbsoluteRegression, ModelModifiedHuber


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
n = 1000
x = np.linspace(-2, 2, n)
ModelClasses = [ModelLinReg, ModelHuber, ModelEpsilonInsensitive,
                ModelAbsoluteRegression]
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
plt.legend(fontsize=13)


plt.subplot(1, 3, 2)
x = np.linspace(-3, 2, n)
ModelClasses = [ModelHinge, ModelModifiedHuber, ModelQuadraticHinge,
                ModelSmoothedHinge, ModelLogReg]

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
plt.legend(fontsize=13)


plt.subplot(1, 3, 3)
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
for model, label in zip(models, labels):
    model.fit(np.array([[1.]]), np.array([1.]))
    y = [model.loss(np.array([t])) for t in x]
    plt.plot(x, y, lw=4, label=label)

plt.xlabel(r"$y'$", fontsize=16)
plt.ylabel(r"$y' \mapsto \ell(1, y')$", fontsize=16)
plt.title('Losses for count data', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13)

plt.tight_layout()
plt.show()
