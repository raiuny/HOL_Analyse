from p_calc import calc_ps_p_formula, calc_uu_p_fsovle, calc_ss_p_fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from conf_calc import calc_conf
def var_lambda1_lambda2_calc_uu_p(n1, n2, tt, tf):
    max_delta_p = 0
    ret1, ret2 = 0, 0 
    for lambda1 in np.arange(0.001, 1/n1, 0.002):
        for lambda2 in np.arange(0.001, 1/n2, 0.002):
            p1, p2, flag = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
            if flag and max(p1, p2) <= 1 and min(p1, p2) > 0:
                delta_p = np.abs(p1 - p2)
                if delta_p > max_delta_p:
                    max_delta_p = delta_p
                    ret1 = lambda1
                    ret2 = lambda2
    return ret1, ret2, max_delta_p

def var_lambda1_lambda2_calc_ss_p(n1, n2, W_1, W_2, K_1, K_2):
    max_delta_p = 0
    ret1, ret2 = 0, 0 
    for lambda1 in np.arange(0.001, 2/n1, 0.002):
        for lambda2 in np.arange(0.001, 2/n2, 0.002):
            p1, p2, flag = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            if flag and max(p1, p2) <= 1 and min(p1, p2) > 0:
                delta_p = np.abs(p1 - p2)
                if delta_p > max_delta_p:
                    max_delta_p = delta_p
                    ret1 = lambda1
                    ret2 = lambda2
    return ret1, ret2, max_delta_p

def var_lambda1_lambda2_calc_ps_p(n1, n2, W_1, W_2, K_1, K_2, tt, tf):
    max_delta_p = 0
    ret1, ret2 = 0, 0 
    for lambda1 in np.arange(0.001, 2/n1, 0.002):
        for lambda2 in np.arange(0.001, 2/n2, 0.002):
            p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
            cf1 = calc_conf(p_us1, p_us2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 1)
            cf2 = calc_conf(p_su1, p_su2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 2)
            idx = np.argmax([cf1, cf2])
            if idx == 0:
                p1, p2 = p_us1, p_us2
            else:
                p1, p2 = p_su1, p_su2   
            delta_p = np.abs(p1 - p2)
            if delta_p > max_delta_p:
                max_delta_p = delta_p
                ret1 = lambda1
                ret2 = lambda2
    return ret1, ret2, max_delta_p

if __name__ =="__main__":
    tt = 36
    tf = 28
    W_1 = 16
    K_1 = 6
    W_2 = 512
    K_2 = 6
    max_delta_p_list = []
    n1_set, n2_set = np.meshgrid(range(10,210,10), range(10,210,10))
    for n1, n2 in zip(n1_set.ravel(), n2_set.ravel()):
        lambda1, lambda2, max_delta_p= var_lambda1_lambda2_calc_ps_p(n1, n2, W_1, W_2, K_1, K_2, tt, tf)
        max_delta_p_list.append(max_delta_p)
    print(np.min(max_delta_p_list), np.max(max_delta_p_list))
    
    def reshape_like(x):
        return np.reshape(x, n1_set.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(n1_set, n2_set , reshape_like(max_delta_p_list), cmap="viridis")
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_zlabel("delta_p of UU")
    plt.show()