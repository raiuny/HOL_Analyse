import matplotlib.pyplot as plt
import numpy as np

tt = 36
tf = 28
lambda2 = 0.036
yp = []
x = np.arange(0.0, 0.99, 0.001)
for p in x:
    z = p - 2 * p * (2 * p - 1) / (2 * p - 1 + 16 * (p - 2 ** 6 * (1 - p) ** 7))
    # print(z)
    yp.append(z)

plt.plot(x, yp, label = "p1 Saturated")
yp2 = []
for p in x:
    z = p - lambda2 * (1 + tf - tf * p - (tt - tf) * p * np.log(p)) / tt
    print(z)
    yp2.append(z)
plt.plot(x, yp2, label = "p2 Unsaturated")
plt.legend()
plt.show()