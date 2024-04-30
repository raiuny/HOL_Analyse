# 2 Link 2 Group MLDs & SLDs
from delay_calc import calc_access_delay_s, calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_formula
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
from conf_calc import calc_conf
tt = 35.8667
tf = 27.2444
n1 = 10 # number of MLDs
n2 = 10 # number of SLDs
# W_1 = 16
K_1 = 6
W_2 = 16
K_2 = 6
lambda_sld = 0.002 * tt
if __name__ == "__main__":
    # 1 link 2 group
    mld_lambda_range = np.arange(0.002, 0.152, 0.002)
    W_mld_range = np.concatenate((np.arange(16, 104, 4), np.arange(128, 512, 64)))
    lambda_set, W_set = np.meshgrid(mld_lambda_range, W_mld_range)
    access_delay_1_ans = []
    access_delay_2_ans = []
    queuing_delay_1_ans = []
    queuing_delay_2_ans = []
    total_delay_1_ans = []
    total_delay_2_ans = []
    throughput_1_ans = []
    throughput_2_ans = []
    p_1_ans = []
    p_2_ans = []
    states = [] # UU: 0 US: 1 SU: 2 SS: 3
    lambda2 = lambda_sld
    for lbd, w in zip(lambda_set.ravel(), W_set.ravel()):
        lambda1 = lbd
        W_1 = w
        p, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
        
        if flag:
            cf = calc_conf(p, p, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 0)
            # print("UU confidence:", cf)
            qd1, ad1 = calc_access_delay_u(p, tt, tf, W_1, K_1, lambda1)
            qd2, ad2 = calc_access_delay_u(p, tt, tf, W_2, K_2, lambda2)
            throughput_1_ans.append(lambda1 * n1)
            throughput_2_ans.append(lambda2 * n2)
            access_delay_1_ans.append(ad1)
            access_delay_2_ans.append(ad2)
            qd1 = qd1 if qd1 > 0 else 1e5
            qd2 = qd2 if qd2 > 0 else 1e5
            queuing_delay_1_ans.append(qd1)
            queuing_delay_2_ans.append(qd2)
            # total_delay_1_ans.append(qd1 + ad1)
            # total_delay_2_ans.append(qd2 + ad2)
            p_1_ans.append(p)
            p_2_ans.append(p)
            states.append(0)
        else:
            p_as, _, flag_ss = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            cf_ss = calc_conf(p_as, p_as, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 3)
            # print("SS confidence:", cf_ss, "p_as:", p_as)
            assert flag_ss is True, "flag ss is False"
            pi_ts_1 = calc_pi_T_S(p_as, tt, tf, W_1, K_1)
            pi_ts_2 = calc_pi_T_S(p_as, tt, tf, W_2, K_2)
            p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
            cf_us, cf_su = 0, 0
            if min(p_us1, p_us2) > 0:
                cf_us = calc_conf(p_us1, p_us2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 1)
            if min(p_su1, p_su2) > 0:
                cf_su = calc_conf(p_su1, p_su2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 2)
            # print("US confidence:", cf_us, "p_us", p_us1, p_us2)
            # print("SU confidence:", cf_su, "p_su", p_su1, p_su2)
            cf_list = [cf_us, cf_su, cf_ss]
            best_idx = np.argmax(cf_list)
            assert np.max(cf_list) != 0
            states.append(best_idx + 1)
            if best_idx == 0: # US
                pi_ts_us1 = calc_pi_T_S(p_us1, tt, tf, W_1, K_1)
                pi_ts_us2 = calc_pi_T_S(p_us2, tt, tf, W_2, K_2)
                throughput_1_ans.append(lambda1 * n1)
                throughput_2_ans.append(min(lambda2, pi_ts_us2) * n2)
                qd1, ad1 = calc_access_delay_u(p_us1, tt, tf, W_1, K_1, lambda1)
                qd2, ad2 = calc_access_delay_s(p_us2, tt, tf, W_2, K_2, lambda2, pi_ts_us2)
                access_delay_1_ans.append(ad1)
                access_delay_2_ans.append(ad2)
                qd1 = qd1 if qd1 > 0 else 1e5
                qd2 = qd2 if qd2 > 0 else 1e5
                queuing_delay_1_ans.append(qd1)
                queuing_delay_2_ans.append(qd2)
                # total_delay_1_ans.append(qd1 + ad1)
                # total_delay_2_ans.append(qd2 + ad2)
                p_1_ans.append(p_us1)
                p_2_ans.append(p_us2)
            elif best_idx == 1: # SU
                pi_ts_su1 = calc_pi_T_S(p_su1, tt, tf, W_1, K_1)
                pi_ts_su2 = calc_pi_T_S(p_su2, tt, tf, W_2, K_2)
                throughput_1_ans.append(min(lambda1, pi_ts_su1) * n1)
                throughput_2_ans.append(lambda2 * n2)
                qd1, ad1 = calc_access_delay_s(p_su1, tt, tf, W_1, K_1, lambda1, pi_ts_su1)
                qd2, ad2 = calc_access_delay_u(p_su2, tt, tf, W_2, K_2, lambda2)
                access_delay_1_ans.append(ad1)
                access_delay_2_ans.append(ad2)
                qd1 = qd1 if qd1 > 0 else 3000
                qd2 = qd2 if qd2 > 0 else 3000
                queuing_delay_1_ans.append(qd1)
                queuing_delay_2_ans.append(qd2)
                # total_delay_1_ans.append(qd1 + ad1)
                # total_delay_2_ans.append(qd2 + ad2)
                p_1_ans.append(p_su1)
                p_2_ans.append(p_su2)
            elif best_idx == 2: # SS
                throughput_1_ans.append(min(lambda1, pi_ts_1) * n1)
                throughput_2_ans.append(min(lambda2, pi_ts_2) * n2)
                qd1, ad1 = calc_access_delay_s(p_as, tt, tf, W_1, K_1, lambda1, pi_ts_1)
                qd2, ad2 = calc_access_delay_s(p_as, tt, tf, W_2, K_2, lambda2, pi_ts_2)
                access_delay_1_ans.append(ad1)
                access_delay_2_ans.append(ad2)
                qd1 = qd1 if qd1 > 0 else 3000
                qd2 = qd2 if qd2 > 0 else 3000
                queuing_delay_1_ans.append(qd1)
                queuing_delay_2_ans.append(qd2)
                # total_delay_1_ans.append(qd1 + ad1)
                # total_delay_2_ans.append(qd2 + ad2)
                p_1_ans.append(p_as)
                p_2_ans.append(p_as)
            
    queuing_delay_1_ans = np.clip(queuing_delay_1_ans, None, 3000)
    queuing_delay_2_ans = np.clip(queuing_delay_2_ans, None, 3000)
    def reshape_like(x):
        return np.reshape(x, lambda_set.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(241, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(states), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('states') 

    ax = fig.add_subplot(242, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(p_1_ans), cmap="viridis")
    ax.plot_surface(lambda_set, W_set , reshape_like(p_2_ans), cmap="coolwarm")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('p of group1 and group2') 

    ax = fig.add_subplot(243, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(access_delay_1_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('access delay of group 1') 
    
    ax = fig.add_subplot(244, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(access_delay_2_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('access delay of group 2') 
    
    ax = fig.add_subplot(245, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(throughput_1_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('throughput of group 1') 
    
    ax = fig.add_subplot(246, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(throughput_2_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('throughput of group 2') 
    
    ax = fig.add_subplot(247, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(queuing_delay_1_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('queuing delay of group 1') 
    
    ax = fig.add_subplot(248, projection="3d")
    ax.plot_surface(lambda_set, W_set , reshape_like(queuing_delay_2_ans), cmap="viridis")
    ax.set_xlabel('lambda_mld')
    ax.set_ylabel('W_mld')
    ax.set_zlabel('queuing delay of group 2') 
    plt.show()
    
    lambda_w_transition_curve = np.load(f"{lambda2}_ss_us_curve0.npy")
    plt.plot(lambda_w_transition_curve[0], lambda_w_transition_curve[1])
    plt.pcolormesh(mld_lambda_range, W_mld_range, reshape_like(states), cmap='viridis')
    plt.colorbar()  # 添加颜色条
    plt.xlabel('group1 lambda')
    plt.ylabel('W')
    plt.title('states')

    plt.show()
    

