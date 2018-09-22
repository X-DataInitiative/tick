import numpy as np

# normalisation the fitting result as f_i(0) = 1
coeff = np.load("sumexplagfall.npy")

U = 4
dim = 2
Total_States = 10

for i in range(dim):
    fi0 = coeff[dim + dim * dim * U + i * Total_States]
    coeff[i] *= fi0
    for u in range(U):
        coeff[dim + dim * dim * u + i * dim: dim + dim * dim * u + (i + 1) * dim] *= fi0
    coeff[dim + dim * dim * U + i * Total_States: dim + dim * dim * U + (i + 1) * Total_States] /= fi0

x_real = np.array(
    [0.4, 0.5,
     0.2, 0.3, 0, 0,
     0.15, 0, 0.2, 0.4,
     0.1, 0.1, 0.2, 0,
     0.1, 0.1, 0, 0.1,
     1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7,
     1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])


print(coeff[:2])
for i in range(U):
    print(coeff[2 + 4 * i : 2 + 4 * (i+1)])
print(coeff[-2 * Total_States:-Total_States])
print(coeff[-Total_States:])
