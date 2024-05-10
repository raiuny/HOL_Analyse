from p_calc import calc_uu_p_fsovle, calc_ss_p_fsolve, calc_ps_p_formula, calc_uu_p_formula
from delay_calc import calc_access_delay_s, calc_access_delay_u
from conf_calc import calc_conf
from state_ana import calc_pi_T_S
import numpy as np
class HOL_Model:
    def __init__(self, n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf) -> None:
        p_uu, _, flag = calc_uu_p_fsovle(n1, lambda1, n2, lambda2, tt, tf)
        p_uu1, _, flag1 = calc_uu_p_formula(n1, lambda1, n2, lambda2, tt, tf)
        print(flag, flag1)
        print(p_uu, p_uu1)
        if flag1:
            qd1, ad1 = calc_access_delay_u(p_uu, tt, tf, W_1, K_1, lambda1)
            qd2, ad2 = calc_access_delay_u(p_uu, tt, tf, W_2, K_2, lambda2)
            self.queuing_delay_1 = qd1
            self.queuing_delay_2 = qd2
            self.access_delay_1 = ad1
            self.access_delay_2 = ad2
            self.p1 = self.p2 = p_uu
            self.state = "UU"
            self.throughput_1 = lambda1 * n1
            self.throughput_2 = lambda2 * n2
        else:
            p_as, flag_ss = calc_ss_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            cf_ss = calc_conf(p_as, p_as, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 3)
            pi_ts_1 = calc_pi_T_S(p_as, tt, tf, W_1, K_1)
            pi_ts_2 = calc_pi_T_S(p_as, tt, tf, W_2, K_2)
            p_us1, p_us2, p_su1, p_su2 = calc_ps_p_formula(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
            cf_us, cf_su = 0, 0
            if min(p_us1, p_us2) > 0:
                cf_us = calc_conf(p_us1, p_us2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 1)
            if min(p_su1, p_su2) > 0:
                cf_su = calc_conf(p_su1, p_su2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, 2)
            cf_list = [cf_us, cf_su, cf_ss]
            assert np.max(cf_list) != 0
            best_idx = np.argmax(cf_list)
            if best_idx == 0: # US
                pi_ts_us1 = calc_pi_T_S(p_us1, tt, tf, W_1, K_1)
                pi_ts_us2 = calc_pi_T_S(p_us2, tt, tf, W_2, K_2)
                self.throughput_1 = lambda1 * n1
                self.throughput_2 = min(lambda2, pi_ts_us2) * n2
                qd1, ad1 = calc_access_delay_u(p_us1, tt, tf, W_1, K_1, lambda1)
                qd2, ad2 = calc_access_delay_s(p_us2, tt, tf, W_2, K_2, lambda2, pi_ts_us2)
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 1e5
                qd2 = qd2 if qd2 > 0 else 1e5
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2

                self.p1 = p_us1
                self.p2 = p_us2
                self.state = "US"
            elif best_idx == 1: # SU
                pi_ts_su1 = calc_pi_T_S(p_su1, tt, tf, W_1, K_1)
                pi_ts_su2 = calc_pi_T_S(p_su2, tt, tf, W_2, K_2)
                self.throughput_1 = min(lambda1, pi_ts_su1) * n1
                self.throughput_2 = lambda2 * n2
                qd1, ad1 = calc_access_delay_s(p_su1, tt, tf, W_1, K_1, lambda1, pi_ts_su1)
                qd2, ad2 = calc_access_delay_u(p_su2, tt, tf, W_2, K_2, lambda2)
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 3000
                qd2 = qd2 if qd2 > 0 else 3000
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2

                self.p1 = p_su1
                self.p2 = p_su2
                self.state = "SU"
            elif best_idx == 2: # SS
                self.throughput_1 = min(lambda1, pi_ts_1) * n1
                self.throughput_2 = min(lambda2, pi_ts_2) * n2
                qd1, ad1 = calc_access_delay_s(p_as, tt, tf, W_1, K_1, lambda1, pi_ts_1)
                qd2, ad2 = calc_access_delay_s(p_as, tt, tf, W_2, K_2, lambda2, pi_ts_2)
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 3000
                qd2 = qd2 if qd2 > 0 else 3000
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2

                self.p1 = p_as
                self.p2 = p_as
                self.state = "SS"
                
    @property
    def p(self):
        return self.p1, self.p2
    
    @property
    def throughput(self):
        return self.throughput_1/32, self.throughput_2/32
    
    @property
    def access_delay(self):
        return self.access_delay_1, self.access_delay_2
    
    @property
    def queuing_delay(self):
        return self.queuing_delay_1, self.queuing_delay_2


if __name__ == "__main__":
    lambda_per_slot = 0.005
    tau_T = 32
    print("lambda_tt: ", tau_T * lambda_per_slot)
    model = HOL_Model(
        n1 = 5,
        n2 = 5,
        lambda1 = tau_T * lambda_per_slot,
        lambda2 = tau_T * lambda_per_slot,
        W_1 = 128,
        W_2 = 128,
        K_1 = 6,
        K_2 = 6,
        tt = tau_T,
        tf = 27
    )
    print("状态: ", model.state)
    print("p: ", model.p)
    print("排队时延: ", model.queuing_delay)
    print("接入时延: ", model.access_delay)
    print("吞吐量: ", model.throughput)


