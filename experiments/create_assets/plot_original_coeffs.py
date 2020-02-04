import numpy as np
import matplotlib.pyplot as plt

from experiments.create_assets.coeffs import plot_coeffs_3_decays

if __name__ == "__main__":
    for version in [1, 2, 3]:
        plot_coeffs_3_decays(np.load(f'../coeffs/original_coeffs_v{version}.npy'))
        plt.savefig(f'../figs/original_coeffs_v{version}.pdf')
