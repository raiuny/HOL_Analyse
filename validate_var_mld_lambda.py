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
n2 = 10 # number of SLDs
beta = [0.5, 0.5]
W_1 = 16
K_1 = 6
W_2 = 16
K_2 = 6

mld_lambda_range = np.arange(0.0001, 0.0031, 0.0001) * tt[0]  # arrival rate per node per slot
sld_lambda = 0.001 * tt[0]

throughput_mld = []
access_delay_mld = []
queuing_delay_mld = []
for lbd in mld_lambda_range:
    lambda1 = lbd
    lambda2 = sld_lambda
    throughput_mld_total = 0
    qd_total = 0
    ad_total = 0
    for i in range(2):
        pL, _, flag = calc_uu_p_formula(n1, lambda1 * beta[i], n2, lambda2, tt[i], tf[i])
        if flag:
            throughput_mld_total += beta[i] * lambda1 * n1
            qd, ad = calc_access_delay_u(pL, tt[i], tf[i], W_1, K_1, lambda1 * beta[i])
            qd_total += beta[i] * qd
            ad_total += beta[i] * ad
        else:
            p_ps = calc_ps_p_fsolve(n1, lambda1*beta[i], n2, lambda2, W_1, K_1, W_2, K_2, tt[i], tf[i])
            p_as, flag2 = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            pi_ts_ps = calc_pi_T_S(p_ps, tt[i], tf[i], W_1, K_1)
            pi_ts_as = calc_pi_T_S(p_as, tt[i], tf[i], W_1, K_1)
            if p_ps > p_as or not flag2: # partial saturated
                if n1 * lambda1 * beta[i] > n2 *lambda2:
                    throughput_mld_total += min(beta[i]*lambda1, pi_ts_ps) * n1
                    _, ad = calc_access_delay_s(p_ps, tt[i], tf[i], W_1, K_1, lambda1 * beta[i], pi_ts_ps)
                    qd_total += 666666
                    ad_total += beta[i] * ad
                else: 
                    throughput_mld_total += beta[i]*lambda1 * n1
                    qd, ad = calc_access_delay_u(p_ps, tt[i], tf[i], W_1, K_1, lambda1 * beta[i])
                    qd_total += qd * beta[i]
                    ad_total += beta[i] * ad
            else: # all saturated
                throughput_mld_total += min(beta[i]*lambda1, pi_ts_as) * n1
                _, ad = calc_access_delay_s(p_ps, tt[i], tf[i], W_1, K_1, lambda1 * beta[i], pi_ts_as)
                qd_total += 666666
                ad_total += beta[i] * ad
    queuing_delay_mld.append(qd_total)
    access_delay_mld.append(ad_total)
    throughput_mld.append(throughput_mld_total)
    
    
print(queuing_delay_mld)
print(access_delay_mld)
print(np.array(throughput_mld) /tt [0] )
plt.plot(mld_lambda_range, throughput_mld)
plt.show()
plt.plot(mld_lambda_range, queuing_delay_mld)
plt.show()
plt.plot(mld_lambda_range, access_delay_mld)
plt.show()