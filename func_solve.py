from scipy.optimize import fsolve, leastsq, root_scalar, root
import numpy as np
import math
from matplotlib import pyplot as plt
lambda1 = 0.00125 * 36
lambda2 = 0.036
tt = 36
tf = 28
n1 = 10
n2 = 10
r = tf/tt
z = r / (1 - (1 - r) * (n1 * lambda1 + n2 * lambda2))
A = (1 + 1/tf) * z 
def x_equation(x):
    return (A * lambda1 * x + 1 - z * lambda1) ** n1 * (A * lambda2 * x + 1 - z * lambda2) ** n2 - x
    

def calc_uu_p(n1, n2, lambda1, lambda2, tt, tf):
    r = tf / tt
    z = r / (1 - (1 - r) * (n1 * lambda1 + n2 * lambda2))
    A = (1 + 1 / tf) * z
    def x_equation(x):
        return (A * lambda1 * x + 1 - z * lambda1) ** n1 * (A * lambda2 * x + 1 - z * lambda2) ** n2 - x
    ans1 = fsolve(x_equation, 0, maxfev=500)[0]
    ans2 = fsolve(x_equation, 5, maxfev=500)[0]
    err1 = x_equation(ans1)
    err2 = x_equation(ans2)
    if err1 > 1e-6:
        return -1, -1, False
    p1 = A * lambda1 + (1 - z * lambda1) * 1 / ans1
    p2 = A * lambda2 + (1 - z * lambda2) * 1 / ans1
    print("p1, p2: ", p1, p2)
    return p1, p2, True

# 使用 fsolve 求解方程
# print(A * lambda1, 1 - z * lambda1, 1 - z * lambda2, A * lambda2)
# x = np.arange(0, 2, 0.0001)
# y = []
# for i in x:
#     y.append(1e9*x_equation(i))
# plt.plot(x, y)
# plt.show()

# ans1 = fsolve(x_equation,  0, xtol = 1e-06, maxfev=500)
# ans2 = fsolve(x_equation,  5, xtol = 1e-06, maxfev=500)
# err1 = x_equation(ans1[0])
# err2 = x_equation(ans2[0])
# print(ans1[0], err1)
# print(ans2[0], err1)
# c = 1/ans1[0]
# p1 = A * lambda1 + (1 - z * lambda1) * c
# p2 = A * lambda2 + (1 - z * lambda2) * c
# print(p1, p2)

# h = leastsq(c_equation, 0.8)
# print(h[0], c_equation(h[0]))
# print(solution, c_equation(solution[0]))
# y = root(c_equation, x0 = 0.8)
# print
# def p(s, l, k, q):
#     p = q * np.maximum(s - k, 0.0)
#     return (p + math.copysign(l, -q)) * math.fabs(q) * 100.0
# result = fsolve(p, 41.0, args=(1,2,3), xtol=1e-06, maxfev=500)
# print(result)

# result = fsolve(p, 41.0, args=(2,3,4), xtol=1e-06, maxfev=500)
# print(result)

def calc_PA2(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
    def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - np.exp(- 2 * nMLD * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) \
                          - 2 * nSLD * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
    pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
    return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)

def calc_alpha_asym(tt, tf, n1, p1, n2, p2):
    n = n1 + n2
    pp = (p1 ** n1 * p2 ** n2) ** (1/(n-1))
    alpha = 1/(tf + 1 + (tt - tf) * (n1 * p1 + n2 * p2 - n * pp) - tf * pp)
    return alpha

def calc_pi_T_S(p1, p2, tt, tf, W, K, ilambda): # ilambda per tt
    alpha = 1 / (1 + tf -tf * p1 - (tt -tf) * p1 * np.log(p1))
    alpha1 = calc_alpha_asym(tt,tf, 10, p1, 10, p2)
    print("alpha", alpha,alpha1)
    p = p1
    ret = 2 * alpha1 * tt * p * (2 * p - 1) / (2 * p - 1 + W * ( p - 2 ** K * (1 - p) ** (K + 1)))
    ret3 = 2 * alpha1 * tt * p * (2 * p - 1) / (W * ( p - 2 ** K * (1 - p) ** (K + 1)))
    # ret3 = 2 * alpha1 * tt /  W * p * (2 * p - 1) / (p - 2 ** K * (1 - p) ** (K + 1))
    print("pi_ts: ", ret, ret3)
    return alpha, alpha1, ret, ret3, ilambda >= ret
lambda2 = 0.0001 * 36
xls = np.arange(0.0001, 0.0015, 0.0001)
p1 = []
p2 = []
pdelta = []
alpha = []
alpha1 = []
pa = []
pi_ts = []
pi_ts1 = []
for x in np.arange(0.0001, 0.0015,0.0001):
    lambda1 = x * 36
    _p1, _p2 , s = calc_uu_p(n1, n2, lambda1, lambda2, tt, tf)
    p1.append(_p1)
    p2.append(_p2)
    a, b, c, d , _= calc_pi_T_S(_p1, _p2, tt, tf, 16, 6, lambda1)
    alpha.append(a)
    alpha1.append(b)
    pi_ts.append(c)
    pi_ts1.append(d)
    _pa = calc_PA2(n1, n2, 16, 6, 16, 6)
    print("pa:", _pa)
    pdelta.append(_p1 - _p2)

plt.plot(xls, pi_ts, label = "pi_ts_true")
plt.plot(xls, pi_ts1, label = "pi_ts1")
plt.legend()
plt.show()
plt.plot(xls, pdelta, label = "p1 - p2")
plt.show()