from state_ana import calc_pi_T_S

def relu(x):
    return x if x > 0 else 0

def calc_conf(p1, p2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, state_idx):
    # UU: 0   US: 1   SU:2   SS: 3
    mu1 = calc_pi_T_S(p1, tt, tf, W_1, K_1)
    mu2 = calc_pi_T_S(p2, tt, tf, W_2, K_2)
    if mu1 < 0 or mu2 < 0:
        return 0
    state_signs = [(1,1), (1,-1), (-1,1), (-1,-1)]
    return 1 - 0.5 * ( relu(state_signs[state_idx][0] * (lambda1 - mu1) ) + relu(state_signs[state_idx][1] * (lambda2 - mu2) ) )
