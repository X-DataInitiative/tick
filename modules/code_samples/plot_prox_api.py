import numpy as np
import matplotlib.pyplot as plt
from tick.prox import ProxL1

x = 0.5 * np.random.randn(50)
a, b = x.min() - 1e-1, x.max() + 1e-1

proxs = [
    ProxL1(strength=0.),
    ProxL1(strength=3e-1),
    ProxL1(strength=3e-1, range=(10, 40)),
    ProxL1(strength=3e-1, positive=True),
    ProxL1(strength=3e-1, range=(10, 40), positive=True),
]

names = [
    "original vector",
    "prox",
    "prox with range=(10, 40)",
    "prox with positive=True",
    "range=(10, 40) and positive=True",
]

_, ax_list = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for prox, name, ax in zip(proxs, names, ax_list):
    ax.stem(prox.call(x))
    ax.set_title(name)
    ax.set_xlim((-1, 51))
    ax.set_ylim((a, b))

plt.tight_layout()
plt.show()