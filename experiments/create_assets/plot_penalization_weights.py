import numpy as np
import matplotlib.pyplot as plt

from experiments.create_assets.coeffs import plot_coeffs_3_decays

if __name__ == "__main__":
    plot_coeffs_3_decays(np.load(f'../coeffs/penalization_weights.npz'))
    plt.savefig(f'../figs/penalization_weights_v1.pdf')
