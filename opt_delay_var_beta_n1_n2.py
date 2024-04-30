import numpy as np
from argparse import ArgumentParser
from p_calc import calc_uu_p_formula, calc_ss_p_fsolve, calc_ps_p_formula
from delay_calc import calc_access_delay_u
from state_ana import calc_pi_T_S
from conf_calc import calc_conf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
lambda_sld = [0.001, 0.001]
sld_num = 200 
n1 = 100
lambda_mld = 0.01
user_ratio = np.arange(0.1, 1.0, 0.1)
W_1 = 8
K_1 = 6
W_2 = 16
K_2 = 6
tt = 35.8667
tf = 27.2444


def delay(n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf):
    p_uu, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
    if flag:
        qd, ad = calc_access_delay_u(p_uu, tt, tf, W_1, K_1, lambda1)
        return ad + 1e4 if qd < 0 else ad + qd
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
            return 2e4
        else:# US
            qd, ad = calc_access_delay_u(p_us1, tt, tf, W_1, K_1, lambda1)
            return ad + 1e4 if qd < 0 else ad + qd

def calc_ST(n1, n2, lambda1, lambda2, tt, tf):
    p_uu, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
    if flag:
        return 0
    else:
        return 1
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-lbd1", "--lambda_mld", default= 0.01, type = float)
    ap.add_argument("-lbd2", "--lambda_sld", default= 0.001, type = float)
    args = ap.parse_args()
    # print(args.lambda_mld,args.lambda_sld)
    lambda_mld = args.lambda_mld
    lambda_sld = [args.lambda_sld]*2
    # opt delay
    best_beta = []
    best_delay = []
    best_states = []
    for r in user_ratio:
        n2 = [sld_num * r, sld_num * (1-r)]
        lambda1 = lambda_mld
        lambda2 = lambda_sld
        def opt_delay(x):
            return x * delay(n1, n2[0], lambda1 * x, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + (1-x) * delay(n1, n2[1], lambda1 * (1-x), lambda2[1], W_1, W_2, K_1, K_2, tt, tf)
        # print(opt_delay(0.5))
        res = minimize(opt_delay, x0 = 0.5)
        if res.fun < 1e4:
            best_beta.append(res.x[0])
            best_delay.append(res.fun)
            best_states.append(calc_ST(n1, n2[0], lambda1 * res.x[0], lambda2[0], tt, tf)*10 + calc_ST(n1, n2[1], lambda1 * (1 - res.x[0]), lambda2[1], tt, tf))
    print(best_delay)
    print(best_beta)
    print("Best States: ", best_states)
    print("User ratio: ", user_ratio)
    
    beta_range = np.arange(0.25, 0.75, 0.002)
    for r in user_ratio:
        n2 = [sld_num * r, sld_num * (1-r)]
        lambda1 = lambda_mld
        lambda2 = lambda_sld
        e2e_delay_list = []
        beta_plot = []
        for x in beta_range:
            e2e_delay = x * delay(n1, n2[0], lambda1 * x, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + (1-x) * delay(n1, n2[1], lambda1 * (1-x), lambda2[1], W_1, W_2, K_1, K_2, tt, tf)
            if e2e_delay < 5000:
                beta_plot.append(x)
                e2e_delay_list.append(e2e_delay)
        plt.plot(beta_plot, e2e_delay_list, label="SLD: n1:n2="+"%.1f" %(r/(1-r)))
    plt.legend()
    plt.xlabel("beta")
    plt.ylabel("E2E delay")
    plt.scatter(best_beta, best_delay, marker="x")
    plt.show()