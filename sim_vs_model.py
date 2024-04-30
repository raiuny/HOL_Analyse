from delay_calc import calc_access_delay_s, calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
tt = [35.8667,35.8667]
tf = [27.2444, 27.2444]
n1 = 10 # number of MLDs
n2 = [10, 10] # number of SLDs
beta = [0.8, 0.2]
W_1 = [16, 16]
K_1 = 6
W_2 = 16
K_2 = 6
lambda_mld_total = 0.002 * tt[0]
lambda_sld = [0.0008 * tt[0], 0.0008*tt[0]]
p_sld2 = [0.965761, 0.964914, 0.964914, 0.963435, 0.962802, 0.963435, 0.963224, 0.963435, 0.962065, 0.965761, 0.962486, 0.963435, 0.963435, 0.962065, 0.963857]
p_sld1 = [0.544275, 0.54889, 0.54889, 0.542629, 0.54982, 0.542629, 0.546655, 0.542629, 0.544915, 0.544275, 0.548628, 0.542629, 0.542629, 0.544915, 0.541712]
p_mld = [0.616876, 0.618735, 0.618735, 0.616857, 0.619354, 0.616857, 0.618907, 0.616857, 0.616866, 0.616876, 0.619249, 0.616857, 0.616857, 0.616866, 0.615831]
p_model_sld = [[],[]]
p_model_mld = [[],[]]
for w in  np.arange(16, 129, 8):
    W_1 = [w, 16]
    for i in range(2):
        lambda1 = lambda_mld_total * beta[i]
        lambda2 = lambda_sld[i]
        
        # sld p calculation
        p_uu, _, flag = calc_uu_p_formula(n1, lambda1, n2[i]-1, lambda2, tt[i], tf[i])
        if flag:
            p_model_sld[i].append(p_uu)
        else:
            p_ps = calc_ps_p_fsolve(n1, lambda1, n2[i], lambda2, W_1[i], K_1, W_2, K_2, tt[i], tf[i])
            p_as, p_as2, flag2= calc_ss_p_fsolve(n1, n2[i], W_1[i],K_1, W_2, K_2)
            if p_ps < p_as2:
                p_model_sld[i].append(p_as2)
            else:
                p_model_sld[i].append(p_ps)
                
        p_uu1, p_uu2, flag3 = calc_uu_p_fsovle(n1, lambda1, n2[i], lambda2, tt[i], tf[i])
        if flag3:
            p_model_mld[i].append(p_uu1)
        else:
            p_ps = calc_ps_p_fsolve(n1, lambda1, n2[i], lambda2, W_1[i], K_1, W_2, K_2, tt[i], tf[i])
            p_as, p_as2, flag2= calc_ss_p_fsolve(n1, n2[i], W_1[i],K_1, W_2, K_2)
            if p_ps < p_as:
                p_model_mld[i].append(p_as)
            else:
                p_model_mld[i].append(p_ps)
    
print(p_model_mld[0], len(p_model_mld[0]))
print(p_model_mld[1], len(p_model_mld[0]))
print(p_model_sld[0], len(p_model_mld[0]))
print(p_model_sld[1], len(p_model_mld[0]))

plt.plot(np.arange(16, 129, 8), p_model_sld[0])
plt.plot(np.arange(16, 129, 8), p_sld1)
plt.show()
plt.plot(np.arange(16, 129, 8), p_model_sld[1])
plt.plot(np.arange(16, 129, 8), p_sld2)
plt.show()
plt.plot(np.arange(16, 129, 8), np.array(p_model_mld[0])*0.8 + np.array(p_model_mld[1])*0.2)
plt.plot(np.arange(16, 129, 8), p_mld)
plt.show()

        