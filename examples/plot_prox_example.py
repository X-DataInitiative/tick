"""
==============================
Examples of proximal operators
==============================

Plot examples of proximal operators available in `tick.prox`
"""

import numpy as np
import matplotlib.pyplot as plt
from tick.prox import ProxL1, ProxElasticNet, ProxL2Sq, \
    ProxPositive, ProxSlope, ProxTV, ProxZero

x = np.random.randn(50)
a, b = x.min() - 1e-1, x.max() + 1e-1
s = 0.4

proxs = [
    ProxZero(),
    ProxPositive(),
    ProxL2Sq(strength=s),
    ProxL1(strength=s),
    ProxElasticNet(strength=s, ratio=0.5),
    ProxSlope(strength=s),
    ProxTV(strength=s)
]

fig, _ = plt.subplots(2, 4, figsize=(16, 8), sharey=True, sharex=True)
fig.axes[0].stem(x)
fig.axes[0].set_title("original vector", fontsize=16)
fig.axes[0].set_xlim((-1, 51))
fig.axes[0].set_ylim((a, b))

for i, prox in enumerate(proxs):
    fig.axes[i + 1].stem(prox.call(x))
    fig.axes[i + 1].set_title(prox.name, fontsize=16)
    fig.axes[i + 1].set_xlim((-1, 51))
    fig.axes[i + 1].set_ylim((a, b))

plt.tight_layout()
plt.show()
