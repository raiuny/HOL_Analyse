import numpy as np
from typing import Tuple


def calc_access_delay_u(p: float, tt: float, tf: float, W: int, K: int, ilambda: float)-> Tuple[float, float]:
    """calculate queuing delay, access delay

    Params:
    ilambda: input rate per node per slot 
    tt: tau_T
    tf: tau_F
    W: length of the initial contend window
    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p)) 
    # alpha = calc_alpha_sym(tt, tf, 20, p)
    # pi_ts, _ = calc_pi_T_S(p, tt, tf, W, K, ilambda * tt)
    alpha = alpha / (1 - ilambda * (1 + tf / tt * (1 - p) / p))
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) + \
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K+1)]
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (3 * alpha ** 2) for i in range(K+1)]
    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret
    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1-p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1

    # according to Geo/G/1's theoretical formula
    ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    queuing_delay = ilambda / tt * (ED0_2 - ED0_1) / (2 * (1 - ilambda / tt * ED0_1))
    access_delay = ED0_1
    # ED01_L = tt + (1 + W) / 2
    # print(ED0_1, ED01_L)
    total_delay = queuing_delay + access_delay
    return queuing_delay, access_delay

def calc_access_delay_s(p: float, tt: float, tf: float, W: int, K: int, ilambda: float, pi_ts: float = 0)-> Tuple[float, float]:
    """calculate queuing delay, access delay

    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    alpha = alpha / (1 - pi_ts * (1 + tf / tt * (1 - p) / p))
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) + \
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K+1)]
    # print("G1Y", G1Y)
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (3 * alpha ** 2) for i in range(K+1)]
    # print("G2Y", G2Y)
    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret
    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1-p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1

    # according to Geo/G/1's theoretical formula
    # ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    # print(ED0_2, ED0_2_L)
    # queuing_delay = ilambda * (ED0_2 - ED0_1) / (2 * (1 - ilambda * ED0_1))
    access_delay = ED0_1
    # ED01_L = tt + (1 + W) / 2
    # print(ED0_1, ED01_L)
    # total_delay = queuing_delay + access_delay
    return -1, access_delay