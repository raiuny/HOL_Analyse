import numpy as np
from argparse import ArgumentParser
from p_calc import calc_uu_p_formula, calc_ss_p_fsolve, calc_ps_p_formula, calc_uu_p_fsovle
from delay_calc import calc_access_delay_u
from state_ana import calc_pi_T_S
from conf_calc import calc_conf
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# sim
n_mld = 10
n_sld_total = 20
n_sld_link1_range = [2, 4, 5, 10] # 1:9, 1:4, 1:3, 1:1
# per-node, per-slot
lambda_mld = 0.002
# per-node, per-slot
lambda_sld1 = 0.0003
lambda_sld2 = 0.0003
beta_range = np.arange(0.45, 0.62, 0.01)
tt = 35.8667
tf = 27.2444

# model
lambda_sld = [lambda_sld1*tt, lambda_sld2*tt]
sld_num = 20
n1 = 10
lambda_mld = lambda_mld * tt
W_1 = 16
K_1 = 6
W_2 = 16
K_2 = 6


def ratio_func(x, a, b):
    return a*x+b

def delay(n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf):
    p_uu, _, flag = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
    if flag:
        qd, ad = calc_access_delay_u(p_uu, tt, tf, W_1, K_1, lambda1)
        return ad + 1e4 if qd < 0 else ad + qd
    else:
        p_as, flag_ss = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
        cf_ss = calc_conf(p_as, p_as, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 3)
        p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
        cf_us, cf_su = 0, 0
        if min(p_us1, p_us2) > 0:
            cf_us = calc_conf(p_us1, p_us2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 1)
        if min(p_su1, p_su2) > 0:
            cf_su = calc_conf(p_su1, p_su2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 2)
        cf_list = [cf_us, cf_su, cf_ss]
        assert np.max(cf_list) != 0
        best_idx = np.argmax(cf_list)
        if best_idx > 0:
            return 2e4
        else:# US
            qd, ad = calc_access_delay_u(p_us1, tt, tf, W_1, K_1, lambda1)
            return ad + 1e4 if qd < 0 else ad + qd


def acc_delay(n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf):
    p_uu, _, flag = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
    if flag:
        qd, ad = calc_access_delay_u(p_uu, tt, tf, W_1, K_1, lambda1)
        return ad
    else:
        p_as, _, flag_ss = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
        cf_ss = calc_conf(p_as, p_as, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 3)
        p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
        cf_us, cf_su = 0, 0
        if min(p_us1, p_us2) > 0:
            cf_us = calc_conf(p_us1, p_us2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 1)
        if min(p_su1, p_su2) > 0:
            cf_su = calc_conf(p_su1, p_su2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 2)
        cf_list = [cf_us, cf_su, cf_ss]
        assert np.max(cf_list) != 0
        best_idx = np.argmax(cf_list)
        if best_idx > 0:
            return 1e4
        else:# US
            qd, ad = calc_access_delay_u(p_us1, tt, tf, W_1, K_1, lambda1)
            return ad

def calc_ST(n1, n2, lambda1, lambda2, tt, tf):
    p_uu, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
    if flag:
        return 0
    else:
        return 1

if __name__ == "__main__":
    # sim 
    data = pd.read_csv("result_e2e_delay.txt")
    print(data)    

    # opt delay
    best_beta = []
    best_delay = []
    best_states = []
    for r in n_sld_link1_range:
        n2 = [r, sld_num - r]
        lambda1 = lambda_mld
        lambda2 = lambda_sld
        def opt_delay(x):
            return x * delay(n1, n2[0], lambda1 * x, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + (1-x) * delay(n1, n2[1], lambda1 * (1-x), lambda2[1], W_1, W_2, K_1, K_2, tt, tf)
        res = minimize(opt_delay, x0 = 0.5,bounds=((0.0001,0.9999),))
        if res.fun < 1e4:
            best_beta.append(res.x[0])
            best_delay.append(res.fun)
            best_states.append(calc_ST(n1, n2[0], lambda1 * res.x[0], lambda2[0], tt, tf)*10 + calc_ST(n1, n2[1], lambda1 * (1 - res.x[0]), lambda2[1], tt, tf))
    print(best_delay)
    print(best_beta)
    print("Best States: ", best_states)
    
    for r in n_sld_link1_range:
        n2 = [r, sld_num - r]
        lambda1 = lambda_mld
        lambda2 = lambda_sld
        e2e_delay_list = []
        beta_plot = []
        for x in beta_range:
            e2e_delay = x * delay(n1, n2[0], lambda1 * x, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + (1-x) * delay(n1, n2[1], lambda1 * (1-x), lambda2[1], W_1, W_2, K_1, K_2, tt, tf)
            if e2e_delay < 5000:
                beta_plot.append(x)
                e2e_delay_list.append(e2e_delay)
        print(f"{r}:{sld_num-r}", e2e_delay_list)
        plt.plot(beta_plot, e2e_delay_list, label=f"SLD: n1:n2={r}:{sld_num-r}")
    len_beta = len(beta_range)
    sim_ed_list = [[],[],[],[]]
    for b, ed1, ed2 in zip(beta_range, data['ed1'][:len_beta], data["ed2"][:len_beta]):
        sim_ed_list[0].append((b * ed1 + (1-b) * ed2)/0.009)
    for b, ed1, ed2 in zip(beta_range, data['ed1'][len_beta:2*len_beta], data["ed2"][len_beta:2*len_beta]):
        sim_ed_list[1].append((b * ed1 + (1-b) * ed2)/0.009)
    for b, ed1, ed2 in zip(beta_range, data['ed1'][2*len_beta:3*len_beta], data["ed2"][2*len_beta:3*len_beta]):
        sim_ed_list[2].append((b * ed1 + (1-b) * ed2)/0.009)
    for b, ed1, ed2 in zip(beta_range, data['ed1'][3*len_beta:], data["ed2"][3*len_beta:]):
        sim_ed_list[3].append((b * ed1 + (1-b) * ed2)/0.009)
    beta1_sim_be1 = np.argmin(sim_ed_list[0])
    beta1_sim_be2 = np.argmin(sim_ed_list[1])
    beta1_sim_be3 = np.argmin(sim_ed_list[2])
    beta1_sim_be4 = np.argmin(sim_ed_list[3])
    plt.scatter(beta_plot[beta1_sim_be1], 54,  label=f"sim SLD: n1:n2={2}:{18}")
    plt.scatter(beta_plot[beta1_sim_be2], 54, label=f"sim SLD: n1:n2={4}:{16}")
    plt.scatter(beta_plot[beta1_sim_be3], 54, label=f"sim SLD: n1:n2={5}:{15}")
    plt.scatter(beta_plot[beta1_sim_be4], 54, label=f"sim SLD: n1:n2={10}:{10}")
    plt.legend()
    plt.xlabel("beta")
    plt.ylabel("E2E delay")
    plt.scatter(best_beta, best_delay, marker="x")
    plt.show()
