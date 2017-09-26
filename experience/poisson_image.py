import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from tick.optim.model import ModelPoisReg
from tick.optim.solver import SDCA
from tick.optim.prox import ProxZero

noise = 300
n_samples = 1000

image = data.camera()
image = image[100:130, 200:240]
image = image.astype(float) * noise / 255.
print(image)

n_features = np.prod(image.shape)

fig, ax_list = plt.subplots(1, 3, figsize=(10, 4))
ax_list[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=0, vmax=noise)

noisy = np.random.poisson(image)

ax_list[1].imshow(noisy, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=0, vmax=noise)


#filters = np.random.rand(n_samples, n_features) * 2 / n_features
filters = np.identity(n_features)
random_filters = np.random.rand(n_samples, n_features) * 2 / n_features
print(filters.shape, random_filters.shape)
filters = np.vstack((filters, random_filters))

photons = np.random.poisson(filters.dot(image.ravel())).astype(float)
print(photons)


l_l2sq = 1e-5
model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(filters, photons)
solver = SDCA(l_l2sq, tol=1e-10)
solver.set_model(model).set_prox(ProxZero())

solver.solve()

reconstructed = solver.solution.reshape(*image.shape)
print(reconstructed, reconstructed.shape)
ax_list[2].imshow(reconstructed, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=reconstructed.min(), vmax=reconstructed.max())

plt.show()