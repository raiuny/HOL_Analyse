# 2 Link 2 Group MLDs & SLDs
from delay_calc import calc_access_delay_s, calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
tt = [35.8667,35.8667]
tf = [27.2444, 27.2444]
n1 = 10 # number of MLDs
n2 = 10 # number of SLDs on each link
beta = [0.5, 0.5]
W_1 = 16
K_1 = 6
W_2 = 16
K_2 = 6
lambda_sld = [0.05, 0.028]

lambda_mld_total = 0.08
lambda_mld_link = [[],[]]
p_link = [[],[]]
e2edelay_link = [[],[]]
queuing_delay_link = [[],[]]
stats = [[], []] # 0: UU 1: US  2: SU 3: SS
for lbd in np.arange(0.01, 0.101, 0.001):
    for i in range(2):
        p, _, flag = calc_uu_p_formula(n1, lbd, n2, lambda_sld[i], tt[i], tf[i])
        if flag and p < 1:
            lambda_mld_link[i].append(lbd)
            qd, ad = calc_access_delay_u(p, tt[i], tf[i], W_1, K_1, lbd)
            if qd < 0:
                qd = -qd
            e2edelay_link[i].append(qd+ad)
            stats[i].append(0)
            p_link[i].append(p)
            queuing_delay_link[i].append(qd)
        else:
            p_as, flag2 = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            p_ps = calc_ps_p_fsolve(n1, lbd, n2, lambda_sld[i], W_1, K_1, W_2, K_2, tt[i], tf[i])
            if p_ps > p_as:
                p_link[i].append(p_ps)
                if lbd * n1 < lambda_sld[i] * n2:
                    lambda_mld_link[i].append(lbd)
                    qd, ad = calc_access_delay_u(p_ps, tt[i], tf[i], W_1, K_1, lbd)
                    print(qd)
                    if qd < 0:
                        qd = -qd
                    e2edelay_link[i].append(qd+ad)
                    queuing_delay_link[i].append(qd)
                    stats[i].append(1)
                else:
                    stats[i].append(2)
            else:
                p_link[i].append(p_as)
                stats[i].append(3)
                continue
print(stats[0])
print(stats[1])
print(p_link[0])
print(p_link[1])
print(lambda_mld_link)
print(e2edelay_link)
fig = plt.figure(1)
ax = fig.add_subplot(321)
ax.plot(lambda_mld_link[0], e2edelay_link[0])
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ylabel('e2e delay on link1')
plt.xlabel('arrival rate on link1')
ax = fig.add_subplot(322)
ax.plot(lambda_mld_link[1], e2edelay_link[1])
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ylabel('e2e delay on link2')
plt.xlabel('arrival rate on link2')


ax = fig.add_subplot(323)
ax.scatter(np.arange(0.01, 0.101, 0.001), p_link[0], marker='+')
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ylabel('p on link1')
plt.xlabel('arrival rate on link1')

ax = fig.add_subplot(324)
ax.scatter(np.arange(0.01, 0.101, 0.001), p_link[1], marker='+')
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ylabel('p on link2')
plt.xlabel('arrival rate on link2')
ax = fig.add_subplot(325)
ax.plot(lambda_mld_link[0], queuing_delay_link[0])
plt.xticks(np.arange(0, 0.10, 0.01))
plt.ylabel('queuing delay on link1')
plt.xlabel('arrival rate on link1')
plt.show()

