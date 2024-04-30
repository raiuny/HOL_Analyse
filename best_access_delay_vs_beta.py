import numpy as np
from argparse import ArgumentParser
from p_calc import calc_uu_p_formula, calc_ss_p_fsolve, calc_ps_p_formula
from delay_calc import calc_access_delay_u
from state_ana import calc_pi_T_S
from conf_calc import calc_conf
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
lambda_sld = [0.001, 0.001]
sld_num = 200
n1 = 100
lambda_mld = 0.01
user_ratio = np.arange(0.1, 1.0, 0.1)
W_1 = 16
K_1 = 6
W_2 = 16
K_2 = 6
tt = 35.8667
tf = 27.2444
def normalize1(x):
    x_f = sum(x)
    return x/x_f

def ratio_func(x, a, b):
    return a*x+b

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


def acc_delay(n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf):
    p_uu, _, flag = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
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
    ap = ArgumentParser()
    ap.add_argument("-lbd1", "--lambda_mld", default= 0.008, type = float)
    ap.add_argument("-lbd2", "--lambda_sld", default= 0.001, type = float)
    args = ap.parse_args()
    # print(args.lambda_mld,args.lambda_sld)
    # opt delay
    best_beta = []
    best_delay = []
    best_states = []
    lambda_mld = args.lambda_mld
    lambda_sld = [args.lambda_sld]*2
    ratio_of_beta_star_list = []
    beta_init_list = []
    for r in np.arange(0, sld_num, 1):
        n2 = [r, 200-r]
        lambda1 = lambda_mld
        lambda2 = lambda_sld
        def opt_delay(x):
            return x * delay(n1, n2[0], lambda1 * x, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + (1-x) * delay(n1, n2[1], lambda1 * (1-x), lambda2[1], W_1, W_2, K_1, K_2, tt, tf)
        # print(opt_delay(0.5))
        res = minimize(opt_delay, x0 = 0.5,bounds=((0.0001,0.9999),))
        if res.fun < 1e4:
            best_beta.append(res.x[0])
            best_delay.append(res.fun)
            best_states.append(calc_ST(n1, n2[0], lambda1 * res.x[0], lambda2[0], tt, tf)*10 + calc_ST(n1, n2[1], lambda1 * (1 - res.x[0]), lambda2[1], tt, tf))
            ratio_of_1_ac = [acc_delay(n1, n2[0], lambda1 * res.x[0], lambda2[0], W_1, W_2, K_1, K_2, tt, tf) * res.x[0], acc_delay(n1, n2[1], lambda1 * (1-res.x[0]), lambda2[1], W_1, W_2, K_1, K_2, tt, tf) *(1-res.x[0])]
            beta_init = ((lambda2[1] * n2[1] - lambda2[0] * n2[0]) / (lambda1 * n1) + 1) / 2
            beta_init = 0.54
            delay_init = [acc_delay(n1, n2[0], lambda1 * beta_init, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) * beta_init ,acc_delay(n1, n2[1], lambda1 * (1-beta_init), lambda2[1], W_1, W_2, K_1, K_2, tt, tf) * (1-beta_init)] 
            delay_init = np.array(delay_init)
            # delay_05 = 0.5 * (delay(n1, n2[0], lambda1 * 0.5, lambda2[0], W_1, W_2, K_1, K_2, tt, tf) + delay(n1, n2[1], lambda1 * (1-beta_init), lambda2[1], W_1, W_2, K_1, K_2, tt, tf) )
            # beta_s = 
            beta_init_list.append(beta_init)
            print(res.x[0], ratio_of_1_ac, delay_init, beta_init)
            ratio_of_1_ac = normalize1(ratio_of_1_ac)
            ratio_of_beta_star_list.append(ratio_of_1_ac[0])
    print(best_delay)
    print(best_beta)
    r1_range = np.arange(0, sld_num,1)/200
    print(ratio_of_beta_star_list)
    print("Best States: ", best_states)
    print("User ratio: ", r1_range)
    params, _ = curve_fit(ratio_func, r1_range, best_beta)
    # x0, y0 = r1_range[0]
    b = (args.lambda_sld * 200 / lambda_mld / 100 + 1)/2
    print("b: ", b)
    print(params, _)
    plt.figure(1)
    plt.scatter(r1_range, best_beta)
    plt.figure(2)
    
    plt.scatter(r1_range, beta_init_list)
    
    plt.show()
