import numpy as np

# normalisation the fitting result as f_i(0) = 1
coeff = np.load("sumexplag.npy")

U = 5
dim = 2
Total_States = 5

for i in range(dim):
    fi0 = coeff[dim + dim * dim * U + i * Total_States]
    coeff[i] *= fi0
    for u in range(U):
        coeff[dim + dim * dim * u + i * dim: dim + dim * dim * u + (i + 1) * dim] *= fi0
    coeff[dim + dim * dim * U + i * Total_States: dim + dim * dim * U + (i + 1) * Total_States] /= fi0

x_real = np.array(
    [0.2, 0.3,  0.0, 0.0, 0.2, 0.0, 0.5,            0.0, 0.3, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.3, 0.0,            0.3, 0.0, 0.0, 0.0, 0.0,

                1., 0.7, 0.8, 0.6, 0.5,				1., 0.6, 0.8, 0.8, 0.6])

x_real = np.array(
    [0.2, 0.3,  0.0, 0.0, 0.0, 0.3,
                0.0, 0.3, 0.0, 0.0,
                0.2, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.3, 0.0,
                0.5, 0.0, 0.0, 0.0,
                1., 0.7, 0.8, 0.6, 0.5,				1., 0.6, 0.8, 0.8, 0.6])

print(coeff[:2])
for i in range(U):
    print(coeff[2 + 4 * i : 2 + 4 * (i+1)])
print(coeff[-10:-5])
print(coeff[-5:])
