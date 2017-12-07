# License: BSD 3 clause

from tick.optim.model import ModelHawkesCustom

import numpy as np

beta = 2.0
MaxN_of_f = 5

timestamps = [np.array([0.31, 0.93, 1.29, 2.32, 4.25, 4.35, 4.78, 5.5, 6.83, 6.99]),
              np.array([0.12, 1.19, 2.12, 2.41, 3.77, 4.21, 4.96, 5.11, 6.7, 7.26])]
T = 9

coeffs = np.array([1., 3., 2., 3., 4., 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
# coeffs = np.array([1., 3.,     2., 3., 4., 1,     1, 3,3,3,3,     2, 4, 6, 8, 10])
# corresponding to mu, alpha,f_i(n)

TestObj = ModelHawkesCustom(beta, MaxN_of_f)
TestObj.fit(timestamps, T)

''' test of the loss function'''
loss_out = TestObj.loss(coeffs)
print(loss_out)

'''test of the grad'''
grad_out = np.array(np.zeros(len(coeffs)))
TestObj.grad(coeffs, grad_out)
print(grad_out)


def custom_loss(coeffs, *argv):
    self = argv[0]
    return self.loss(coeffs)


def custom_grad(coeffs, *argv):
    self = argv[0]
    grad_out = np.array(np.zeros(len(coeffs)))
    self.grad(coeffs, grad_out)
    return grad_out


print(custom_loss(coeffs, TestObj))
print(custom_grad(coeffs, TestObj))

print('#' * 40, '\nTest of gradient\n', '#' * 40)
# print(check_grad(custom_loss, custom_grad, coeffs, TestObj))

manual_grad = []
dup_coeffs = coeffs.copy()
epsilon = 1e-7
for i in range(len(coeffs)):
    dup_coeffs[i] += epsilon
    loss_out_new = custom_loss(dup_coeffs, TestObj)
    manual_grad.append((loss_out_new - loss_out) / epsilon)
    dup_coeffs[i] -= epsilon
manual_grad = np.array(manual_grad)
print(manual_grad)

###########################################################################################
import tick

print(tick)
