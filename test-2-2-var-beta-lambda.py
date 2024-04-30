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
n2 = [10, 10] # number of SLDs
beta = []
W_1 = [16, 16]
K_1 = 6
W_2 = 16
K_2 = 6
# lambda_mld_total = 0.08
lambda_sld = [0.03, 0.03]
if __name__ == "__main__":
    # test 1 adjust beta to achieve the Minimum access delay
    beta_range = np.arange(0.01, 1.0, 0.01)
    lambda_mld_range = np.arange(0.01, 1.01, 0.01)
    beta_set, lambda_set = np.meshgrid(beta_range, lambda_mld_range)
    access_delay_ans = []
    throughput_1_ans = []
    throughput_2_ans_link1 = []
    throughput_2_ans_link2 = []
    throughput_total_ans = []
    total_delay_ans = []
    p_1_ans= []
    for b, lbd in zip(beta_set.ravel(), lambda_set.ravel()):
        beta = [b, 1-b]
        lambda_mld_total = lbd / n1 # per node per tt
        throughput_1 = 0
        throughput_2 = [0, 0]
        throughput_total = 0
        access_delay = 0
        p_1 = 0
        for i in range(2):
            lambda1 = lambda_mld_total * beta[i]
            lambda2 = lambda_sld[i]
            p, _, flag = calc_uu_p_formula(n1, lambda1, n2[i], lambda2, tt[i], tf[i])
            if flag:
                _, ad = calc_access_delay_u(p, tt[i], tf[i], W_1[i], K_1, lambda1)
                throughput_1 += lambda1 * n1 
                throughput_2[i] = lambda2 * n2[i]
                p_1 += beta[i] * p
                access_delay += beta[i] * ad
            else:
                p_as, flag_ss = calc_ss_p_fsolve(n1, n2[i], W_1[i], K_1, W_2, K_2)
                assert flag_ss is True, "flag ss is False"
                pi_ts_1 = calc_pi_T_S(p_as, tt[i], tf[i], W_1[i], K_1)
                pi_ts_2 = calc_pi_T_S(p_as, tt[i], tf[i], W_2, K_2)
                p_ps = calc_ps_p_fsolve(n1, lambda1, n2[i], lambda2, W_1[i], K_1, W_2, K_2, tt[i], tf[i])
                pi_ts_ps_1 = calc_pi_T_S(p_ps, tt[i], tf[i], W_1[i], K_1)
                pi_ts_ps_2 = calc_pi_T_S(p_ps, tt[i], tf[i], W_2, K_2)
                if p_ps > p_as: # partial saturated
                    p_1 += beta[i] * p_ps
                    if lambda1 * n1 > lambda2 * n2[i]:
                        throughput_1 += min(pi_ts_ps_1, lambda1) * n1
                        throughput_2[i] = lambda2 * n2[i]
                        _, ad = calc_access_delay_s(p_ps, tt[i], tf[i], W_1[i], K_1, lambda1, pi_ts_ps_1)
                        access_delay += beta[i] * ad
                    else:
                        throughput_1 += lambda1 * n1
                        throughput_2[i] = min(pi_ts_ps_2, lambda2) * n2[i]
                        _, ad = calc_access_delay_u(p_ps, tt[i], tf[i], W_1[i], K_1, lambda1)
                        access_delay += beta[i] * ad
                else: # all saturated
                    p_1 += beta[i] * p_as
                    throughput_1 += min(pi_ts_1, lambda1) * n1
                    throughput_2[i] = min(pi_ts_2, lambda2) * n2[i]
                    _, ad = calc_access_delay_s(p_as, tt[i], tf[i], W_1[i], K_1, lambda1, pi_ts_1)
                    access_delay += beta[i] * ad
        access_delay_ans.append(access_delay)
        throughput_1_ans.append(throughput_1)
        throughput_2_ans_link1.append(throughput_2[0])
        throughput_2_ans_link2.append(throughput_2[1])
        p_1_ans.append(p_1)
    
    def reshape_like(x):
        return np.reshape(x, beta_set.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(221, projection="3d")
    # ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="coolwarm")
    ax.plot_surface(beta_set, lambda_set , reshape_like(p_1_ans), cmap="viridis")
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda_mld')
    ax.set_zlabel('p')    
    
    ax = fig.add_subplot(222, projection="3d")
    ax.plot_surface(beta_set, lambda_set , reshape_like(throughput_1_ans), cmap="viridis")
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda_mld')
    ax.set_zlabel('throughput_mld')   
    
    ax = fig.add_subplot(223, projection="3d")
    ax.plot_surface(beta_set, lambda_set , reshape_like(throughput_2_ans_link1), cmap="viridis")
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda_mld')
    ax.set_zlabel('throughput_sld_link1')  
    
    ax = fig.add_subplot(224, projection="3d")
    ax.plot_surface(beta_set, lambda_set , reshape_like(throughput_2_ans_link2), cmap="viridis")
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda_mld')
    ax.set_zlabel('throughput_sld_link2')  
    plt.show()
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="coolwarm")
    ax.plot_surface(beta_set, lambda_set , reshape_like(access_delay_ans), cmap="viridis")
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda_mld')
    ax.set_zlabel('access delay of MLD')  
    plt.show()