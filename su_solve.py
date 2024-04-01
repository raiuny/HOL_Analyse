from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# x = np.arange(0.3,0.99999, 0.001)
# # x = np.arange(0.001,0.4999, 0.00001)
# y = []
# y2 = []
# for p in x:
#     z = (p - 2 ** 6 * (1 - p) ** 7) / (2 * p - 1)
#     z2 = 7 - 6 * p
#     print(z)
#     y.append(z)
#     y2.append(z2)

# plt.plot(x, y)
# plt.plot(x, y2)
# plt.show()

lambda1 = 0.002 * 36
lambda2 = 0.036
tt = 36
tf = 28
n1 = 10
n2 = 10
r = tf/tt
W1 = 16
K1 = 6
def p1_func(p1):
    c = p1 - 2 * (2 * p1 - 1) / (2 * p1 - 1 + W1 * (p1 - 2 ** K1 * (1 - p1) ** (K1+1)))
    p2 = lambda2 * (r + 1 / tt - r * c + (1 - r) * n1 * (p1 - c)) / (1 - n2 * lambda2 * (1 - r)) + c
    return c ** (n1 + n2 - 1) - p1 ** n1 * p2 ** n2

p1 = fsolve(p1_func, 1.2)[0]
err1 = p1_func(p1)
c = p1 - 2 * (2 * p1 - 1) / (2 * p1 - 1 + W1 * (p1 - 2 ** K1 * (1 - p1) ** (K1+1)))
p2 = lambda2 * (r + 1 / tt - r * c + (1 - r) * n1 * (p1 - c)) / (1 - n2 * lambda2 * (1 - r)) + c
print(p1, err1, p2, c)
