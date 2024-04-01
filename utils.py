# from scipy.special import lambertw
# import numpy as np
# from typing import List, Tuple
# from math import exp
# from scipy.optimize import root_scalar, root
#
# def calc_uu_p(nmld: int, mld_ilambda: float, nsld: int, sld_lambda: float, tt: float, tf: float)-> Tuple[float, float, float, bool]:
#     """ calculate p in U-U scenario, return p of each link (both unsaturated)
#     Args:
#         nmld (int): number of MLDs
#         nsld (int): number of SLDs
#         mld_ilambda: lambda of each link
#         tt: tau_T
#         tf: tau_F
#     Returns:
#         p, throughput of MLD, uu status or not
#     """
#     mld_ilambda = mld_ilambda * tt
#     sld_lambda = sld_lambda * tt
#     uu = True
#     mld_thpt = nmld * mld_ilambda
#     sld_thpt = nsld * sld_lambda
#     allambda = mld_thpt + sld_thpt
#     A = (allambda * tf / tt) / (1 - (1 - tf / tt) * allambda)
#     B = -(allambda * (1 + tf) / tt) / (1 - (1 - tf / tt) * allambda)
#     pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
#     # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
#     if not np.isreal(lambertw(B*exp(-A), -1)):
#         uu = False
#         # print(f"Warn: saturated!")
#     return pL, mld_thpt, sld_thpt, uu
#
# def calc_su_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, float, float, bool]:
#     """ calculate p in S-U scenario, return p of each link (MLD unsaturated, SLD saturated)
#     Args:
#         nmld (int): number of MLDs
#         nsld (int): number of SLDs
#         mld_ilambda: lambda of each link
#         tt: tau_T
#         tf: tau_F
#     Returns:
#         p, throughput, su status or not
#     """
#     mld_lambda = mld_lambda * tt
#     sld_lambda = sld_lambda * tt
#     a_sld_lambda = nsld * sld_lambda
#
#     A = (a_sld_lambda * tf / tt - 4 * nmld / W_mld) / (1 - (1 - tf / tt) * a_sld_lambda)
#     B = (-a_sld_lambda * (1 + tf) / tt + 2 * nmld / W_mld) / (1 - (1 - tf / tt) * a_sld_lambda)
#     pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
#     # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
#     # print(pL, pS, lambertw(B*exp(-A)))
#     # if not np.isreal(lambertw(B*exp(-A), -1)):
#     #     su = False
#     #     print(f"Warn: saturated!")
#     # pL, err = calc_psu(nmld, nsld, mld_lambda, tt, tf, W_sld, K_sld)
#     pi_ts, a = calc_pi_T_S(pL, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
#     pi_ts_, a_ = calc_pi_T_S(pL, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
#     # print(pi_ts, a, pi_ts_, a_)
#     if a and not a_:
#         su = True
#     else:
#         su = False
#     # print(pi_ts, a)
#     thpt = a_sld_lambda + nmld * pi_ts
#     return pL, nmld * pi_ts, a_sld_lambda, su
#
# def calc_us_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, float, float, bool]:
#     """ calculate p in U-S scenario, return p of each link (MLD unsaturated, SLD saturated)
#     Args:
#         M (int): number of links
#         nmld (int): number of MLDs
#         nsld (List): numbers of SLDs for each link
#         kk: kappa of MLDs
#         tt: tau_T
#         tf: tau_F
#     Returns:
#         p on each link, throughput of each link
#     """
#     mld_lambda = mld_lambda * tt
#     sld_lambda = sld_lambda * tt
#     a_mld_lambda = nmld * mld_lambda
#
#     A = (a_mld_lambda * tf / tt - 4 * nsld / W_sld) / (1 - (1 - tf / tt) * a_mld_lambda)
#     B = (-a_mld_lambda * (1 + tf) / tt + 2 * nsld / W_sld) / (1 - (1 - tf / tt) * a_mld_lambda)
#     pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
#     # pL = B * np.real(lambertw(B*exp(-A), 0))
#     # print("pL=", pL)
#     # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
#     # print(pL, pS, lambertw(B*exp(-A)))
#     pi_ts, a = calc_pi_T_S(pL, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
#     pi_ts_, a_ = calc_pi_T_S(pL, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
#     if a and not a_:
#         us = True
#     else:
#         us = False
#     thpt = a_mld_lambda + nsld * pi_ts
#     return pL, a_mld_lambda, nsld * pi_ts, us
#
# def calc_ss_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, float, float, bool]:
#     """ calculate p in S-S scenario, return p of each link and throughput on each link (both saturated)
#     Args:
#         M (int): number of links
#         nmld (int): number of MLDs
#         nsld (List): numbers of SLDs for each link
#         kk: kappa of MLDs
#         tt: tau_T
#         tf: tau_F
#     Returns:
#         p on each link, throughput on each link
#     """
#     mld_lambda = mld_lambda * tt
#     sld_lambda = sld_lambda * tt
#     pa, err = calc_PA2(nmld, nsld, W_mld, K_mld, W_sld, K_sld)
#     # pa, err = calc_PA(nmld, nsld, W_mld, K_mld, W_sld, K_sld)
#     # print(err)
#     pi_ts, a = calc_pi_T_S(pa, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
#     pi_ts_, a_ = calc_pi_T_S(pa, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
#     if a and a_:
#         ss = True
#     else:
#         ss = False
#     thpt = nmld * pi_ts + nsld * pi_ts_
#     return pa, nmld * pi_ts, nsld * pi_ts_, ss
#
# def calc_pi_T_S(p, tt, tf, W, K, ilambda):
#     # ilambda = ilambda * tt
#     alpha = 1 / (1 + tf -tf * p - (tt -tf) * p * np.log(p))
#     # # alpha = 1 / (1 + tf + (tt - tf) * (40*p - 40*p**(40/39))-tf*p**(40/39))
#     # # print("alpha", alpha1,alpha)
#     ret = 2 * alpha * tt / W * p * (2 * p - 1) / (p - 2 ** K * (1 - p) ** (K + 1))
#     # A = alpha * tt
#     # B  = sum([(1 - p) ** i * (W * 2 ** i + 1)  for i in range(K)]) + (1 - p) ** K / p * (1 + W * 2 ** K)
#     # ret2  = A / B
#     return ret, ilambda >= ret
#
# # def calc_pi_T_U(p, tt, tf, W, K, ilambda):
# #     ilambda = ilambda * tt
# #     pi_ts, _ = calc_pi_T_S(p, tt, tf, W, K, ilambda)
# #     ret = pi_ts / (1 + (1 + tf / tt * (1 - p) / p) * (pi_ts - ilambda))
# #     return ret, ilambda < ret
#
# def calc_access_delay(p: float, tt: float, tf: float, W: int, K: int, ilambda: float)-> Tuple[float, float]:
#     """calculate queuing delay, access delay
#
#     Returns:
#         Tuple[float, float]: queuing delay, access delay
#     """
#     alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
#     ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) + \
#         W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (p * (2 * p - 1))))
#     G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K+1)]
#     # print("G1Y", G1Y)
#     G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (3 * alpha ** 2) for i in range(K+1)]
#     # print("G2Y", G2Y)
#     def sum_iK(i):
#         ret = 0
#         for j in range(i, K):
#             ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
#         return ret
#     # calculate G''D0(1)
#     A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
#     B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
#     C = (1 - p) ** K / p * G2Y[K]
#     D = (1 - p) ** K / p * G1Y[K]
#     G2D01 = A + C + 2 * (p * tt + (1-p) * tf) * (B + D) + 2 * \
#         sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in range(K)])\
#             + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (1 - p) / p * (tf ** 2 - tf)
#     ED0_2 = G2D01 + ED0_1
#
#     # according to Geo/G/1's theoretical formula
#     ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
#     # print(ED0_2, ED0_2_L)
#     queuing_delay = ilambda * (ED0_2 - ED0_1) / (2 * (1 - ilambda * ED0_1))
#     access_delay = ED0_1
#     # ED01_L = tt + (1 + W) / 2
#     # print(ED0_1, ED01_L)
#     total_delay = queuing_delay + access_delay
#     return queuing_delay, access_delay
#
#
# def get_status(p, tt, tf, W, K, ilambda):
#     ilambda = ilambda * tt
#     pi_ts, a = calc_pi_T_S(p, tt, tf, W, K, ilambda)
#     if(a):
#         return "S"
#     else:
#         return "U"
#
# def p_func(p, n, w, k):
#     if k == np.inf:
#         p_fn = p - np.exp((-2 * n) * (2 * p - 1) / (p * w))
#     else:
#         p_fn = p - np.exp((-2 * n) * (2 * p - 1) / ( (2 * p - 1) + w * (p - 2 ** k * (1 - p) ** (k + 1))))
#     return p_fn
#
# def calc_PA(n, w, k):
#     pa = root_scalar(p_func, args=(n, w, k), bracket=[0.00001, 0.99999], method='brentq').root
#     return pa, p_func(pa, n, w, k)
#
# def calc_PA2(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
#     def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
#         return p - np.exp(- 2 * nMLD * (2 * p - 1) / (W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) \
#             - 2 * nSLD * (2 * p - 1) / (W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
#     pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
#     return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)
#
# def calc_psu(nMLD, nSLD, mld_lambda, tt, tf, W_sld, K_sld):
#     def psuf(p, nMLD, nSLD, mld_lambda, tt, tf):
#         return p - np.exp(- (nMLD * mld_lambda * (1 + tf - tf * p - (tt - tf) * p * np.log(p))) / (tt * p) - 2 * nSLD * (2 * p - 1) / (W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
#     pb = root_scalar(psuf, args=(nMLD, nSLD, mld_lambda, tt, tf), bracket=[0.1, 0.99999], method='brentq').root
#     return pb, psuf(pb, nMLD, nSLD, mld_lambda, tt, tf)

from scipy.special import lambertw
import numpy as np
from typing import List, Tuple
from math import exp
from scipy.optimize import root_scalar, fsolve

def calc_uu_p_fsovle(n1, n2, lambda1, lambda2, tt, tf):
    r = tf / tt
    z = r / (1 - (1 - r) * (n1 * lambda1 + n2 * lambda2))
    A = (1 + 1 / tf) * z
    def x_equation(x):
        return (A * lambda1 * x + 1 - z * lambda1) ** n1 * (A * lambda2 * x + 1 - z * lambda2) ** n2 - x
    ans1 = fsolve(x_equation, 0, maxfev=500)[0]
    ans2 = fsolve(x_equation, 5, maxfev=500)[0]
    err1 = x_equation(ans1)
    err2 = x_equation(ans2)
    if err1 > 1e-6:
        return -1, -1, False
    p1 = A * lambda1 + (1 - z * lambda1) * 1 / ans1
    p2 = A * lambda2 + (1 - z * lambda2) * 1 / ans1
    print("p1, p2: ", p1, p2)
    return p1, p2, True

def calc_uu_p(nmld: int, mld_ilambda: float, nsld: int, sld_lambda: float, tt: float, tf: float)-> Tuple[float, float, float, bool]:
    """ calculate p in U-U scenario, return p of each link (both unsaturated)
    Args:
        nmld (int): number of MLDs
        nsld (int): number of SLDs
        mld_ilambda: lambda of each link
        tt: tau_T
        tf: tau_F
    Returns:
        p, throughput of MLD, uu status or not
    """
    mld_ilambda = mld_ilambda * tt
    sld_lambda = sld_lambda * tt
    uu = True
    allambda = nmld * mld_ilambda + nsld * sld_lambda
    A = (allambda * tf / tt) / (1 - (1 - tf / tt) * allambda)
    B = -(allambda * (1 + tf) / tt) / (1 - (1 - tf / tt) * allambda)
    pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
    pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
    pL2 = B / np.real(lambertw(B*exp(-A), 0))
    print("uu pL, pL2", pL, pL2)
    p1, p2, flag = calc_uu_p_fsovle(nmld, nsld, mld_ilambda, sld_lambda, tt, tf)
    if mld_ilambda == 0.0025 / 2 * 36:
        pi_ts, a = calc_pi_T_S(pL2, tt, tf, 16, 6, mld_ilambda) # 判断mld是否饱和
        pi_ts_, a_ = calc_pi_T_S(pL2, tt, tf, 16, 6, sld_lambda) # 判断sld是否饱和
    if not np.isreal(lambertw(B*exp(-A), -1)):
        uu = False
        # print("saturated!")
    return pL2,pL, nmld * mld_ilambda, nsld * sld_lambda, uu

def calc_su_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int, psu0)-> Tuple[float, float, float, bool]:
    """ calculate p in S-U scenario, return p of each link (MLD unsaturated, SLD saturated)
    Args:
        nmld (int): number of MLDs
        nsld (int): number of SLDs
        mld_ilambda: lambda of each link
        tt: tau_T
        tf: tau_F
    Returns:
        p, throughput, su status or not
    """
    mld_lambda = mld_lambda * tt
    sld_lambda = sld_lambda * tt
    a_sld_lambda = nsld * sld_lambda
    psu = psu0
    p_init = 0.0001
    while(True):
        try:
            if p_init >= 1:
                break
            psu, err = calc_psu(nmld, nsld, sld_lambda, W_mld, K_mld, tt, tf, p_init)
            print("p_init, psu, err", p_init, psu, err)
            break
        except Exception as e:
            p_init += 0.0001
    
    # nmld = nmld - 1
    # psu = calc_psu(nmld, nsld, mld_lambda, tt, tf, W_sld, K_sld)
    A = (a_sld_lambda * tf / tt - 4 * (nmld) / W_mld) / (1 - (1 - tf / tt) * a_sld_lambda)
    B = (-a_sld_lambda * (1 + tf) / tt + 2 * (nmld) / W_mld) / (1 - (1 - tf / tt) * a_sld_lambda)
    pL = B / np.real(lambertw(B*exp(-A), 0))
    pS = B / np.real(lambertw(B*exp(-A), -1))

    # print("su psu pL, pS", psu, pL, pS, np.isreal(lambertw(B*exp(-A), -1)))
    # 
    err = psu_func(pL, nmld, nsld, mld_lambda, W_sld, K_sld, tt, tf)
    print("su err", err)
    # err2 = psu_func(0.52, nmld, nsld, mld_lambda, W_sld, K_sld, tt, tf)
    # print("su err2", err2)
    # print(calc_psu(nmld, nsld, mld_lambda, W_sld, K_sld,  tt, tf,pL2-0.2))
    # print(pL, pS, lambertw(B*exp(-A)))
    # if not np.isreal(lambertw(B*exp(-A), -1)):
    #     su = False
    #     print(f"Warn: saturated!")
    # p_su = calc_psu(nmld, nsld, mld_lambda, tt, tf, W_sld, K_sld)
    # print("psu   psu", pS, pL, p_su)
    # pL = 0.52
    pi_ts, a = calc_pi_T_S(psu, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
    # mld_lambda = 0.03
    # for p in np.arange(0.481, 0.7, 0.01):
    #     pi_ts, a = calc_pi_T_S(p, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
    #     if (pi_ts  > 0.00125 * 36):
    #         print("*************************")
    #     print(pi_ts * nmld / tt)
    pi_ts_, a_ = calc_pi_T_S(psu, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
    # print("su pi_ts", pi_ts, a, pi_ts_, a_)
    # if not np.isreal(lambertw(B*exp(-A), -1)):
    #     a = True
    if a and not a_:
        su = True
    else:
        su = False
    # print(pi_ts, a)
    if pi_ts > mld_lambda:
        return psu, nmld * mld_lambda, a_sld_lambda, su
    else:
        return psu, nmld * pi_ts, a_sld_lambda, su
    # thpt = a_sld_lambda + nmld * pi_ts


def calc_us_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, float, float, bool]:
    """ calculate p in U-S scenario, return p of each link (MLD unsaturated, SLD saturated)
    Args:
        M (int): number of links
        nmld (int): number of MLDs
        nsld (List): numbers of SLDs for each link
        kk: kappa of MLDs
        tt: tau_T
        tf: tau_F
    Returns:
        p on each link, throughput of each link
    """
    mld_lambda = mld_lambda * tt
    sld_lambda = sld_lambda * tt
    a_mld_lambda = nmld * mld_lambda

    A = (a_mld_lambda * tf / tt - 4 * nsld / W_sld) / (1 - (1 - tf / tt) * a_mld_lambda)
    B = (-a_mld_lambda * (1 + tf) / tt + 2 * nsld / W_sld) / (1 - (1 - tf / tt) * a_mld_lambda)
    # pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
    # pL = B * np.real(lambertw(B*exp(-A), 0))
    # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
    # print(pL, pS, lambertw(B*exp(-A)))
    pL = B / np.real(lambertw(B*exp(-A), 0))
    # print("us pL, pL2", pL, pL2)
    pi_ts, a = calc_pi_T_S(pL, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
    pi_ts_, a_ = calc_pi_T_S(pL, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
    if a and not a_:
        us = True
    else:
        us = False
    thpt = a_mld_lambda + nsld * pi_ts
    return pL, a_mld_lambda, nsld * pi_ts, us

def calc_ss_p(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, float, float, bool]:
    """ calculate p in S-S scenario, return p of each link and throughput on each link (both saturated)
    Args:
        M (int): number of links
        nmld (int): number of MLDs
        nsld (List): numbers of SLDs for each link
        kk: kappa of MLDs
        tt: tau_T
        tf: tau_F
    Returns:
        p on each link, throughput on each link
    """
    mld_lambda = mld_lambda * tt
    sld_lambda = sld_lambda * tt
    pa1, err1 = calc_PA2(nmld-1, nsld, W_mld, K_mld, W_sld, K_sld)
    pa2, err2 = calc_PA2(nmld, nsld-1, W_mld, K_mld, W_sld, K_sld)
    print("pa, pa2", pa1, pa2)
    print("err", err1, err2)
    
    A = -4 * (nmld / W_mld + nsld / W_sld)
    B = 2 * (nmld / W_mld + nsld / W_sld)
    pL = B / np.real(lambertw(B*exp(-A), 0 ))
    print(B*exp(-A) > -1/np.exp(1), pL)
    # pa1, err1 = calc_PA2(nmld-1, nsld, W_mld, K_mld, W_sld, K_sld)
    # pa2, err2 = calc_PA2(nmld, nsld-1, W_mld, K_mld, W_sld, K_sld)
    # print("pa, pa2", pa1, pa2)
    # print("err", err1, err2)
    pi_ts, a = calc_pi_T_S(pL, tt, tf, W_mld, K_mld, mld_lambda) # 判断mld是否饱和
    pi_ts_, a_ = calc_pi_T_S(pL, tt, tf, W_sld, K_sld, sld_lambda) # 判断sld是否饱和
    return pa1, pa2, pi_ts, pi_ts_

def calc_pi_T_S(p, tt, tf, W, K, ilambda): # ilambda per tt
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    # alpha = calc_alpha_sym(tt, tf, 20, p)
    # A = alpha * tt
    # B = sum([(1 - p) ** i / 2 * (1 + W * 2 ** i) for i in range(K)]) + (1 - p) ** K / p * (1 + W * 2 ** K) / 2
    # ret2 = A / B
    ret = 2 * alpha * tt * p * (2 * p - 1) / (2 * p - 1 + W * ( p - 2 ** K * (1 - p) ** (K + 1)))
    # print(ret, ret2)
    return ret, ilambda >= ret

def calc_pi_T_U(p, tt, tf, W, K, ilambda): # ilambda per tt
    pi_ts, _ = calc_pi_T_S(p, tt, tf, W, K, ilambda)
    ret = pi_ts / (1 + (1 + tf / tt * (1 - p) / p) * (pi_ts - ilambda))
    return ret, ilambda < ret

def calc_access_delay_u(p: float, tt: float, tf: float, W: int, K: int, ilambda: float)-> Tuple[float, float]:
    """calculate queuing delay, access delay

    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p)) 
    # alpha = calc_alpha_sym(tt, tf, 20, p)
    # pi_ts, _ = calc_pi_T_S(p, tt, tf, W, K, ilambda * tt)
    alpha = alpha / (1 - ilambda * tt * (1 + tf / tt * (1 - p) / p))
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
    ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    # print(ED0_2, ED0_2_L)
    queuing_delay = ilambda * (ED0_2 - ED0_1) / (2 * (1 - ilambda * ED0_1))
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
    # pi_ts, _ = calc_pi_T_S(p, tt, tf, W, K, ilambda * tt)
    # print(pi_ts, ilambda*tt)
    alpha = alpha / (1 - ilambda * tt * (1 + tf / tt * (1 - p) / p))
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
    ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    # print(ED0_2, ED0_2_L)
    queuing_delay = ilambda * (ED0_2 - ED0_1) / (2 * (1 - ilambda * ED0_1))
    access_delay = ED0_1
    # ED01_L = tt + (1 + W) / 2
    # print(ED0_1, ED01_L)
    total_delay = queuing_delay + access_delay
    return queuing_delay, access_delay

def get_status(p, tt, tf, W, K, ilambda):
    ilambda = ilambda * tt
    pi_ts, a = calc_pi_T_S(p, tt, tf, W, K, ilambda)
    if(a):
        return "S"
    else:
        return "U"

def p_func(p, n, w, k):
    if k == np.inf:
        p_fn = p - np.exp((-2 * n) * (2 * p - 1) / (p * w))
    else:
        p_fn = p - np.exp((-2 * n) * (2 * p - 1) / ( (2 * p - 1) + w * (p - 2 ** k * (1 - p) ** (k + 1))))
    return p_fn

def calc_PA1(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
    def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) ) ** nMLD *\
                        (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1)))) ** nSLD
    pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
    return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)

def calc_PA2(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
    def pf(p, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld):
        return p - np.exp(- 2 * nMLD * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))) \
                          - 2 * nSLD * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
    pa = root_scalar(pf, args=(nMLD, nSLD, W_mld, K_mld, W_sld, K_sld), bracket=[0.00001, 0.99999], method='brentq').root
    return pa, pf(pa, nMLD, nSLD, W_mld, K_mld, W_sld, K_sld)


def psu_func(p, nMLD, nSLD, sld_lambda, W_mld, K_mld, tt, tf):
    return p - exp(- nSLD * sld_lambda * (1 + tf - tf * p - (tt - tf) * p * np.log(p)) / (tt * p) - 2 * nMLD * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))))
def psu_func2(p, nMLD, nSLD, sld_lambda, W_mld, K_mld, tt, tf):
    return p - np.exp(-(nSLD * sld_lambda * (1 + tf)) / (tt - nSLD * sld_lambda * (tt -tf)) / p + nSLD * sld_lambda * tf / (tt - nSLD * sld_lambda * (tt - tf)) - 2 * nMLD * (2 * p - 1) / (2 * p - 1 + W_mld * (p - 2 ** K_mld * (1 - p) ** (K_mld + 1))))
def calc_psu(nMLD, nSLD, sld_lambda, W_mld, K_mld, tt, tf, psu_init):
    # print(nMLD, nSLD, mld_lambda, tt, tf)
    pb = root_scalar(psu_func2, args=(nMLD, nSLD, sld_lambda, W_mld, K_mld, tt, tf), bracket=[psu_init, 0.99999], method='brentq').root
    return pb, psu_func(pb, nMLD, nSLD, sld_lambda, W_sld, K_sld, tt, tf)


def pus_func(p, nMLD, nSLD, mld_lambda, W_sld, K_sld, tt, tf):
    return p - np.exp(-(nMLD * mld_lambda * (1 + tf)) / (tt - nMLD * mld_lambda * (tt -tf)) / p + nMLD * mld_lambda * tf / (tt - nMLD * mld_lambda * (tt - tf)) - 2 * nSLD * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))

def calc_pus(nMLD, nSLD, mld_lambda, W_sld, K_sld, tt, tf, psu_init):
    # print(nMLD, nSLD, mld_lambda, tt, tf)
    pb = root_scalar(pus_func, args=(nMLD, nSLD, mld_lambda, W_sld, K_sld, tt, tf), bracket=[psu_init, 0.99999], method='brentq').root
    return pb, psu_func(pb, nMLD, nSLD, mld_lambda, W_sld, K_sld, tt, tf)



def calc_alpha_sym(tt, tf, n, p): 
    alpha = 1/(tf + 1 + (tt - tf) * (n * p - n * p ** (n/(n-1))) - tf * p ** (n/(n-1)))
    return alpha

if __name__ == "__main__":
    # p, err = calc_psu(10,10,0.025, 36,28,16,6)
    # print(p)
    nMLD = 10
    nSLD = 10
    mld_lambda = 0.054
    tt = 36
    tf = 28
    W_sld = 16
    K_sld = 6
    A = (mld_lambda * tf / tt - 4 * nSLD / W_sld) / (1 - (1 - tf / tt) * mld_lambda)
    B = (-mld_lambda * (1 + tf) / tt + 2 * nSLD / W_sld) / (1 - (1 - tf / tt) * mld_lambda)
    pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
    print("pL", pL)
    def psuf(p, nMLD, nSLD, mld_lambda, tt, tf):
        return p - np.exp(- (nMLD * mld_lambda * (1 + tf - tf * p - (tt - tf) * p * np.log(p))) / (tt * p) - 2 * nSLD * (2 * p - 1) / (2 * p - 1 + W_sld * (p - 2 ** K_sld * (1 - p) ** (K_sld + 1))))
    kk = 0
    for p in np.arange(0.03, 0.9, 0.001):
        kk  = psuf(p, 10,10,0.054,36,28)
        if kk < 0:
            print(p)
    pss1, thrpt, _, b= calc_ss_p(10, 0.0015, nSLD, 0.001, tt, tf, 16, 6, 16, 6)
    print(b)
    pss2, _, _, _= calc_ss_p(10, 0.0015, nSLD-1, 0.001, tt, tf, 16, 6, 16, 6)
    # puu, _, _, _ = calc_su_p(10, 0.0014, nSLD, 0.001, tt, tf)
    print(pL, kk, pss1, pss2, _)
    print(calc_access_delay_u(pL, tt, tf, 16, 6, mld_lambda/36))
    print(calc_access_delay_u(kk, tt, tf, 16, 6, mld_lambda/36))
    print(calc_access_delay_u(pss1, tt, tf, 16, 6, mld_lambda/36))
    print(calc_access_delay_u(pss2, tt, tf, 16, 6, mld_lambda/36))
    print(thrpt)
    