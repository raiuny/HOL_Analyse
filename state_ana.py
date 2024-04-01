import numpy as np

# 节点饱和时的服务率 mu_S
def calc_pi_T_S(p: float, tt: float, tf: float, W: int, K: int) -> float:
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    ret = 2 * alpha * tt * p * (2 * p - 1) / (2 * p - 1 + W * ( p - 2 ** K * (1 - p) ** (K + 1)))
    return ret

# 节点未饱和时的服务率 mu_U
def calc_pi_T_U(p: float, tt: float, tf: float, W: int, K: int, lambda1: float) -> float:
    pi_ts = calc_pi_T_S(p, tt, tf, W, K)
    ret = pi_ts / (1 + (1 + tf / tt * (1 - p) / p) * (pi_ts - lambda1))
    return ret

def analyse_state(lambda1, p, tt, tf, W, K) -> int:
    """_summary_
    Returns:
        int: 1 (Saturated) , 0 (Unsaturated) , -1 (Unknown)
    """
    pi_tu = calc_pi_T_U(p, tt, tf, W, K, lambda1)
    pi_ts = calc_pi_T_S(p, tt, tf, W, K)
    if pi_ts > lambda1 and lambda1 > pi_tu:
        # print(f"I don't know the state of this node {lambda1}")
        return -1
    if pi_ts < lambda1:
        return 1
    if pi_tu > lambda1:
        return 0
    return -1
    # return pi_ts , pi_tu

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lambda1_range = np.arange(0.030, 0.05, 0.0002)
    p_range = np.arange(0.001, 0.999,0.0002)
    tt = 36
    tf = 28
    W = 64
    K = 6
    x_s = []
    p_s = []
    x_u = []
    p_u = []
    for p in p_range:
        for x in lambda1_range:
            state = analyse_state(x, p, tt, tf, W, K)
            if state == 1:
                x_s.append(x)
                p_s.append(p)
            elif state == 0:
                x_u.append(x)
                p_u.append(p)               
    plt.scatter(x_s, p_s, c='red', label="Saturated")
    plt.scatter(x_u, p_u, c='green', label="Unsaturated")
    plt.xlabel("lambda")
    plt.ylabel("p")
    plt.legend()
    plt.show()
    