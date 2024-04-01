from delay_calc import calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
W = 16
K = 6
n1 = 10 # number of MLDs
n2 = 10 # number of SLDs
tt = 36
tf = 28
if __name__ == "__main__":
    lambda1_range = np.arange(0.001, 0.05, 0.001)
    lambda2_range = np.arange(0.001, 0.05, 0.001)
    lambda1_set, lambda2_set = np.meshgrid(lambda1_range, lambda2_range)
    access_delay_ans = []
    access_delay_ans2 = []
    p_ans = []
    p_ans2 = []
    queuing_delay_ans = []
    queuing_delay_ans2 = []
    err_ac_delay = []
    for lambda1, lambda2 in zip(lambda1_set.ravel(), lambda2_set.ravel()):
        pL, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
        pL2, _, flag2 = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
        # ld = calc_pi_T_S(pL2, tt, tf, W, K)
        if not flag or not flag2:
            d1, d2 = analyse_state(lambda1, pL2, tt, tf, W, K)
            d11, d22 = analyse_state(lambda2, pL2, tt, tf, W, K)
            print(lambda1, lambda2, "#", d1, d2, lambda1*n1+lambda2*n2,  flag, flag2, sep='\t')
        # print(lambda1, lambda2, pL)
        qd, ad = calc_access_delay_u(pL, tt, tf, W, K, lambda1)
        qd2, ad2 = calc_access_delay_u(pL2, tt, tf, W, K, lambda1)
        p_ans.append(pL)
        p_ans2.append(pL2)
        queuing_delay_ans.append(min(1000,max(qd, 0)))
        queuing_delay_ans2.append(min(1000,max(qd, 0)))
        access_delay_ans.append(ad)
        access_delay_ans2.append(ad2)
        err_ac_delay.append(ad-ad2)
    # print(access_delay_ans)
    # print(err_ac_delay)
    
    def reshape_like(x):
        return np.reshape(x, lambda1_set.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(221, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans), cmap="coolwarm")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(p_ans2), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('p')    
    # plt.show()
    ax = fig.add_subplot(222, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(queuing_delay_ans), cmap="coolwarm")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(queuing_delay_ans2), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('queuing delay') 
    ax = fig.add_subplot(223, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(access_delay_ans), cmap="coolwarm")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(access_delay_ans2), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('access delay') 
    ax = fig.add_subplot(224, projection="3d")
    ax.plot_surface(lambda1_set, lambda2_set, reshape_like(err_ac_delay), cmap="coolwarm")
    # ax.plot_surface(lambda1_range, lambda2_range, reshape_like(access_delay_ans2), cmap="viridis")
    ax.set_xlabel('lambda1')
    ax.set_ylabel('lambda2')
    ax.set_zlabel('err_ac_delay') 
    plt.show()
