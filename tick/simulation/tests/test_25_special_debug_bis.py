# License: BSD 3 clause

import numpy as np

dim = 2
MaxN_of_f = 3
U = 4

def fmufalpha(coeff):
    fmu = []
    falpha = []
    for k in range(dim):
        mu = coeff[k]
        alpha = []
        for u in range(U):
            alpha.append(coeff[dim + u * dim * dim + k * 2: dim + u * dim * dim + (k + 1) * 2])
        alpha = np.concatenate(alpha).ravel()

        f = coeff[dim + U * dim * dim + k * MaxN_of_f: dim + U * dim * dim + (k + 1) * MaxN_of_f]

        fmu.append(np.array(mu * f))
        falpha.append(np.array(mu * alpha))

    return fmu, falpha



tmp1 = np.load("real.npy")
tmp2 = np.load("fit.npy")

fmu1, falpha1 = fmufalpha(tmp1)
fmu2, falpha2 = fmufalpha(tmp2)

for k in range(dim):
    print('-' * 60)
    print("dim : ", k)
    print(fmu2[k] /fmu1[k])
    print(falpha2[k] /falpha1[k])