import matplotlib.pyplot as plt
from tick.simulation import weights_sparse_exp, weights_sparse_gauss

n_features = 100
weights1 = weights_sparse_exp(n_features, nnz=20)
weights2 = weights_sparse_gauss(n_features, nnz=10)

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.stem(weights1)
plt.subplot(1, 2, 2)
plt.stem(weights2)

plt.tight_layout()
plt.show()