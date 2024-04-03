# 1-2-as: 1 Link 2 Group All-Saturated
from delay_calc import calc_access_delay_s, calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
W = 64
K = 6
n1 = 10 # number of MLDs
n2 = 10 # number of SLDs
tt = 36
tf = 28
if __name__ == "__main__":
    lambda1_range = np.arange(0.035, 0.045, 0.0002)
    lambda2_range = np.arange(0.035, 0.045, 0.0002)
    lambda1_set, lambda2_set = np.meshgrid(lambda1_range, lambda2_range)
    access_delay_ans = []
    access_delay_ans2 = []
    throughput_ans = []
    p_ans = []
    p_ans2 = []
    err_ac_delay = []
    p_as, _= calc_ss_p_fsolve(n1, n2, W, K, W, K)
    pi_ts = calc_pi_T_S(p_as, tt, tf, W, K)
    throughput_ans_1 = []
    for lambda1, lambda2 in zip(lambda1_set.ravel(), lambda2_set.ravel()):
        p_au, _, flag_uu = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
        # pL2, _, flag4 = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
        # p, flag_ss= calc_ss_p_formula(n1, lambda1, n2, lambda2, tt, tf, W, K, W, K)
        # ld = calc_pi_T_S(pL2, tt, tf, W, K)
        if flag_uu:
            throughput_ans.append(lambda1 * n1 + lambda2 * n2)
            throughput_ans_1.append(lambda1 * n1)
            p_ans.append(p_au)
            _, ad = calc_access_delay_u(p_au, tt, tf, W, K, lambda1)
            access_delay_ans.append(ad)
        else:
            p_ps = calc_ps_p_fsolve(n1, lambda1, n2, lambda2, W, K, W, K, tt, tf)
            pi_ts_ps = calc_pi_T_S(p_ps, tt, tf, W, K)
            print(lambda1, lambda2, lambda1*n1 + lambda2 *n2, p_ps, pi_ts_ps, pi_ts, pi_ts_ps > pi_ts, p_ps > p_as, sep="\t")
            if p_ps > p_as: # partial saturated
                p_ans.append(p_ps)
                throughput_ans.append(min(pi_ts_ps, lambda1) * n1 + min(pi_ts_ps, lambda2) * n2)
                throughput_ans_1.append(min(pi_ts_ps, lambda1) * n1)
                if lambda1 * n1 > lambda2 * n2:
                    _, ad = calc_access_delay_s(p_ps, tt, tf, W, K, lambda1, pi_ts_ps)
                else:
                    _, ad = calc_access_delay_u(p_ps, tt, tf, W, K, lambda1)
                access_delay_ans.append(ad)
            else: # all saturated
                p_ans.append(p_as)
                throughput_ans.append(min(pi_ts, lambda1) * n1 + min(pi_ts, lambda2) * n2)
                throughput_ans_1.append(min(pi_ts, lambda1) * n1)
                _, ad = calc_access_delay_s(p_as, tt, tf, W, K, lambda1, pi_ts)
                access_delay_ans.append(ad)
        # _, ad = calc_access_delay_s(p, tt, tf, W, K, lambda1, pi_ts)
        # _, ad2 = calc_access_delay_s(p2, tt, tf, W, K, lambda1)
        # access_delay_ans.append(ad)
        # access_delay_ans2.append(ad2)
        # err_ac_delay.append(ad-ad2)
    # print(access_delay_ans)
    # print(err_ac_delay)
    # print(access_delay_ans)
    def reshape_like(x):
        return np.reshape(x, lambda1_set.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(221, projection="3d")
    # ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="coolwarm")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('p')    
    ax = fig.add_subplot(222, projection="3d")
    # ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="coolwarm")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(access_delay_ans), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('access_delay')  
     
    ax = fig.add_subplot(223, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(throughput_ans), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('total throughput')  
    
    ax = fig.add_subplot(224, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(throughput_ans_1), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('total throughput MLD')  
    plt.show()