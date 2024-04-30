# 2 Link 2 Group MLDs & SLDs
from delay_calc import calc_access_delay_s, calc_access_delay_u
from p_calc import calc_uu_p_formula, calc_uu_p_fsovle, calc_ss_p_formula, calc_ss_p_fsolve, calc_ps_p_fsolve, calc_ps_p_formula
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from state_ana import calc_pi_T_S, analyse_state
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
from conf_calc import calc_conf

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# input
# fix data rate, number of STAs, traffic arrival rate, backoff cutoff stage
tt = [35.8667, 35.8667]
tf = [27.2444, 27.2444]
n_mld = 10
n_sld_total = 20
n_sld_link1_range = [4, 5, 10] # 1:4, 1:3, 1:1
# per-node, per-slot
lambda_mld = 0.002
# per-node, per-slot
lambda_sld1 = 0.001
lambda_sld2 = 0.001
K_mld = [6, 6]
K_sld = [6, 6]
beta_range = np.arange(0.1, 0.91, 0.04)
W_mld = [16, 16]
W_sld = [16, 16]
n_sld_link1_set, beta_set = np.meshgrid(n_sld_link1_range, beta_range)

# output
ana_status_perlink = np.zeros((*beta_set.shape, 2)) # status number
STAT_UNDEF, STAT_UU, STAT_US, STAT_SU, STAT_SS = (0, 1, 2, 3, 4)
ana_p_mld_perlink = np.zeros((*beta_set.shape, 2))
ana_p_sld_perlink = np.zeros((*beta_set.shape, 2))
ana_p_mld_total = np.zeros(beta_set.shape)
ana_tp_mld_perlink = np.zeros((*beta_set.shape, 2))
ana_tp_mld_total = np.zeros(beta_set.shape)
ana_tp_sld_perlink = np.zeros((*beta_set.shape, 2))
ana_que_mld_perlink = np.zeros((*beta_set.shape, 2))
ana_que_mld_total = np.zeros(beta_set.shape)
ana_que_sld_perlink = np.zeros((*beta_set.shape, 2))
ana_acc_mld_perlink = np.zeros((*beta_set.shape, 2))
ana_acc_mld_total = np.zeros(beta_set.shape)
ana_acc_sld_perlink = np.zeros((*beta_set.shape, 2))
ana_e2e_mld_perlink = np.zeros((*beta_set.shape, 2))
ana_e2e_mld_total = np.zeros(beta_set.shape)
ana_e2e_sld_perlink = np.zeros((*beta_set.shape, 2))


if __name__ == "__main__":
    # iterate through rectangular grid of beta and SLD1's lambda
    for mesh_index, nsld in np.ndenumerate(n_sld_link1_set):
        i, j = mesh_index
        n_sld = [nsld, n_sld_total-nsld]
        lambda_sld = [lambda_sld1, lambda_sld2]
        beta = [beta_set[i][j], 1-beta_set[i][j]]

        # per-link analysis
        for l in range(2):
            lamb_mld = lambda_mld * beta[l] * tt[l]
            lamb_sld = lambda_sld[l] * tt[l]
            p, psps, flag = calc_uu_p_formula(n_mld, lamb_mld, n_sld[l], lamb_sld, tt[l], tf[l])
            if flag:
                # Status is UU
                cf = calc_conf(p, p, lamb_mld, lamb_sld, W_mld[l], K_mld[l], W_sld[l], K_sld[l], tt[l], tf[l], 0)
                qd_mld, ad_mld = calc_access_delay_u(p, tt[l], tf[l], W_mld[l], K_mld[l], lamb_mld)
                qd_sld, ad_sld = calc_access_delay_u(p, tt[l], tf[l], W_sld[l], K_sld[l], lamb_sld)

                ana_status_perlink[i][j][l] = STAT_UU
                ana_p_mld_perlink[i][j][l] = p
                ana_p_sld_perlink[i][j][l] = p
                ana_tp_mld_perlink[i][j][l] = lamb_mld * n_mld
                ana_tp_sld_perlink[i][j][l] = lamb_sld * n_sld[l]
                ana_que_mld_perlink[i][j][l] = qd_mld
                ana_que_sld_perlink[i][j][l] = qd_sld
                ana_acc_mld_perlink[i][j][l] = ad_mld
                ana_acc_sld_perlink[i][j][l] = ad_sld
                ana_e2e_mld_perlink[i][j][l] = ana_que_mld_perlink[i][j][l] + ana_acc_mld_perlink[i][j][l]
                ana_e2e_sld_perlink[i][j][l] = ana_que_sld_perlink[i][j][l] + ana_acc_sld_perlink[i][j][l]

            else:
                # Status is US, SU, or SS
                p_as, flag_ss = calc_ss_p_fsolve(n_mld, n_sld[l], W_mld[l], K_mld[l], W_sld[l], K_sld[l])
                cf_ss = calc_conf(p_as, p_as, lamb_mld, lamb_sld, W_mld[l], K_mld[l], W_sld[l], K_sld[l], tt[l], tf[l], 3)
                assert flag_ss is True, "falg ss is false"
                p_us_mld, p_us_sld, p_su_mld, p_su_sld = calc_ps_p_formula(n_mld, lamb_mld, n_sld[l], lamb_sld, W_mld[l], K_mld[l], W_sld[l], K_sld[l], tt[l], tf[l])
                cf_us, cf_su = 0, 0
                if min(p_us_mld, p_us_sld) > 0:
                    cf_us = calc_conf(p_us_mld, p_us_sld, lamb_mld, lamb_sld, W_mld[l], K_mld[l], W_sld[l], K_sld[l], tt[l], tf[l], 1)
                if min(p_su_mld, p_su_sld) > 0:
                    cf_su = calc_conf(p_su_mld, p_su_sld, lamb_mld, lamb_sld, W_mld[l], K_mld[l], W_sld[l], K_sld[l], tt[l], tf[l], 2)
                cf_list = [cf_us, cf_su, cf_ss]
                best_idx = np.argmax(cf_list)
                if np.max(cf_list) == 0:
                    assert False
                else:
                    ana_status_perlink[i][j][l] = best_idx + 2

                # Get corresponding results
                if ana_status_perlink[i][j][l] == STAT_US:
                    pi_ts_ps_mld = calc_pi_T_S(p_us_mld, tt[l], tf[l], W_mld[l], K_mld[l])
                    pi_ts_ps_sld = calc_pi_T_S(p_us_sld, tt[l], tf[l], W_sld[l], K_sld[l])
                    qd_mld, ad_mld = calc_access_delay_u(p_us_mld, tt[l], tf[l], W_mld[l], K_mld[l], lamb_mld)
                    _, ad_sld = calc_access_delay_s(p_us_sld, tt[l], tf[l], W_sld[l], K_sld[l], lamb_sld, pi_ts_ps_sld)

                    ana_p_mld_perlink[i][j][l] = p_us_mld
                    ana_p_sld_perlink[i][j][l] = p_us_sld
                    ana_tp_mld_perlink[i][j][l] = lamb_mld * n_mld
                    ana_tp_sld_perlink[i][j][l] = min(pi_ts_ps_sld, lamb_sld) * n_sld[l]
                    ana_que_mld_perlink[i][j][l] = qd_mld
                    ana_que_sld_perlink[i][j][l] = np.inf
                    ana_acc_mld_perlink[i][j][l] = ad_mld
                    ana_acc_sld_perlink[i][j][l] = ad_sld
                    ana_e2e_mld_perlink[i][j][l] = ana_que_mld_perlink[i][j][l] + ana_acc_mld_perlink[i][j][l]
                    ana_e2e_sld_perlink[i][j][l] = ana_que_sld_perlink[i][j][l] + ana_acc_sld_perlink[i][j][l]

                if ana_status_perlink[i][j][l] == STAT_SU:
                    pi_ts_ps_mld = calc_pi_T_S(p_su_mld, tt[l], tf[l], W_mld[l], K_mld[l])
                    pi_ts_ps_sld = calc_pi_T_S(p_su_sld, tt[l], tf[l], W_sld[l], K_sld[l])
                    _, ad_mld = calc_access_delay_s(p_su_mld, tt[l], tf[l], W_mld[l], K_mld[l], lamb_mld, pi_ts_ps_mld)
                    qd_sld, ad_sld = calc_access_delay_u(p_su_sld, tt[l], tf[l], W_sld[l], K_sld[l], lamb_sld)

                    ana_p_mld_perlink[i][j][l] = p_su_mld
                    ana_p_sld_perlink[i][j][l] = p_su_sld
                    ana_tp_mld_perlink[i][j][l] = min(pi_ts_ps_mld, lamb_mld) * n_mld
                    ana_tp_sld_perlink[i][j][l] = lamb_sld * n_sld[l]
                    ana_que_mld_perlink[i][j][l] = np.inf
                    ana_que_sld_perlink[i][j][l] = qd_sld
                    ana_acc_mld_perlink[i][j][l] = ad_mld
                    ana_acc_sld_perlink[i][j][l] = ad_sld
                    ana_e2e_mld_perlink[i][j][l] = ana_que_mld_perlink[i][j][l] + ana_acc_mld_perlink[i][j][l]
                    ana_e2e_sld_perlink[i][j][l] = ana_que_sld_perlink[i][j][l] + ana_acc_sld_perlink[i][j][l]

                if ana_status_perlink[i][j][l] == STAT_SS:
                    pi_ts_as_mld = calc_pi_T_S(p_as, tt[l], tf[l], W_mld[l], K_mld[l])
                    pi_ts_as_sld = calc_pi_T_S(p_as, tt[l], tf[l], W_sld[l], K_sld[l])
                    _, ad_mld = calc_access_delay_s(p_as, tt[l], tf[l], W_mld[l], K_mld[l], lamb_mld, pi_ts_as_mld)
                    _, ad_sld = calc_access_delay_s(p_as, tt[l], tf[l], W_sld[l], K_sld[l], lamb_sld, pi_ts_as_sld)

                    ana_p_mld_perlink[i][j][l] = p_as
                    ana_tp_mld_perlink[i][j][l] = min(pi_ts_as_mld, lamb_mld) * n_mld
                    ana_tp_sld_perlink[i][j][l] = min(pi_ts_as_sld, lamb_sld) * n_sld[l]
                    ana_que_mld_perlink[i][j][l] = np.inf
                    ana_que_sld_perlink[i][j][l] = np.inf
                    ana_acc_mld_perlink[i][j][l] = ad_mld
                    ana_acc_sld_perlink[i][j][l] = ad_sld
                    ana_e2e_mld_perlink[i][j][l] = ana_que_mld_perlink[i][j][l] + ana_acc_mld_perlink[i][j][l]
                    ana_e2e_sld_perlink[i][j][l] = ana_que_sld_perlink[i][j][l] + ana_acc_sld_perlink[i][j][l]

            if ana_que_mld_perlink[i][j][l] > 1000 or ana_que_mld_perlink[i][j][l] < -1000:
                ana_que_mld_perlink[i][j][l] = np.nan

            if ana_p_mld_perlink[i][j][l] >= 1:
                print(f'i={i}, j={j}, l={l}, stat={ana_status_perlink[i][j][l]}, p={ana_p_mld_perlink[i][j][l]}')

            if ana_p_sld_perlink[i][j][l] >= 1:
                print(f'i={i}, j={j}, l={l}, stat={ana_status_perlink[i][j][l]}, p={ana_p_sld_perlink[i][j][l]}')

        # calculate total results
        ana_p_mld_total[i][j] = np.dot(ana_p_mld_perlink[i][j], beta)
        ana_tp_mld_total[i][j] = np.sum(ana_tp_mld_perlink[i][j])
        ana_que_mld_total[i][j] = np.dot(ana_que_mld_perlink[i][j], beta)
        ana_acc_mld_total[i][j] = np.dot(ana_acc_mld_perlink[i][j], beta)
        ana_e2e_mld_total[i][j] = np.dot(ana_e2e_mld_perlink[i][j], beta)

    assert np.min(ana_status_perlink) != 0

    print('n1:n2=1:4')
    print("ana_que_mld_total", ','.join(str(item) for item in ana_que_mld_total[:, 0]))
    print("ana_acc_mld_total", ','.join(str(item) for item in ana_acc_mld_total[:, 0]))
    print("ana_e2e_mld_total", ','.join(str(item) for item in ana_e2e_mld_total[:, 0]))

    print('\nn1:n2=1:3')
    print("ana_que_mld_total", ','.join(str(item) for item in ana_que_mld_total[:, 1]))
    print("ana_acc_mld_total", ','.join(str(item) for item in ana_acc_mld_total[:, 1]))
    print("ana_e2e_mld_total", ','.join(str(item) for item in ana_e2e_mld_total[:, 1]))

    print('\nn1:n2=1:1')
    print("ana_que_mld_total", ','.join(str(item) for item in ana_que_mld_total[:, 2]))
    print("ana_acc_mld_total", ','.join(str(item) for item in ana_acc_mld_total[:, 2]))
    print("ana_e2e_mld_total", ','.join(str(item) for item in ana_e2e_mld_total[:, 2]))
