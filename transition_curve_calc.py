from state_ana import calc_pi_T_S
import numpy as np
from p_calc import calc_ss_p_fsolve, calc_uu_p_formula
def calc_transition_curve_lambda_w_ss_us(n1, n2, lambda2, W_2, K_1, K_2, tt, tf):
    def p_ss_func(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) ) ** nMLD *\
                        (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1)))) ** nSLD
    def p_ps_func(p, n_s, W_s, K_s, n_u, lambda_u):
        return p - np.exp(-(n_u * lambda_u) * (1 + tf - tf * p - (tt - tf) * np.log(p) * p) / (tt * p) - 2 * n_s * (2 * p - 1) / (2 * p - 1 + W_s * (p - 2 ** K_s * (1 - p) ** (K_s + 1))))
    # group1 from S to U
    w_list = []
    lambda1_list = []
    lambda_uu = 0
    w_uu = 4
    for w in np.arange(4, 516, 4):
        p_ss,_, flag = calc_ss_p_fsolve(n1, n2, w, K_1, W_2, K_2)
        lambda1 = calc_pi_T_S(p_ss, tt, tf, w, K_1)
        pi_ts_2 = calc_pi_T_S(p_ss, tt, tf, W_2, K_2)
        p_uu, _, flag2 = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
        if pi_ts_2 < lambda2 and flag and not flag2:
            w_list.append(w)
            lambda1_list.append(lambda1)
        if flag2:
            if lambda_uu < lambda1:
                lambda_uu = lambda1
                w_uu = w

    w_list = np.concatenate((w_list, np.arange(w_uu, 516, 4)))
    lambda1_list = np.concatenate((lambda1_list,[lambda_uu]*len(np.arange(w_uu,516,4))))
    print(w_list)
    print(lambda1_list)
    return w_list, lambda1_list

# def transition_curve_lambda_w_ss_us(n1, n2, lambda2, W_2, K_1, K_2, tt, tf):
    
# def calc_transition_curve_lambda_w_ss_uu(n1, n2, lambda2, W_2, K_1, K_2, tt, tf):

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n1 = 10
    n2 = 10
    tt = 35.8667
    tf = 27.2444
    lambda2 = 0.002 * tt
    W_2 = 16
    K_1 = 6
    K_2 = 6
    wl, ll = calc_transition_curve_lambda_w_ss_us(n1, n2, lambda2, W_2, K_1, K_2, tt, tf)
    data = np.array([ll, wl])
    print(data)
    
    np.save(f"{lambda2}_ss_us_curve.npy", data)
    data2 = np.load(f"{lambda2}_ss_us_curve.npy")
    print(data2[0])
    plt.plot(ll, wl)
    plt.show()
    
    