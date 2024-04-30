from scipy.optimize import fsolve, root_scalar
from scipy.special import lambertw
from typing import Tuple
from math import exp
import numpy as np

def calc_uu_p_fsovle(n1: int, lambda1: float, n2: int, lambda2: float, tt: float, tf: float)-> Tuple[float, float, bool]:
    """_summary_

    Args:
        n1 (int): number of mlds
        lambda1 (float): mld's lambda
        n2 (int): number of slds
        lambda2 (float): sld's lambda
        tt (float): tau_T
        tf (float): tau_F

    Returns:
        Tuple[float, float, bool]: pL, pS, is_correct
    """
    r = tf / tt
    z = r / (1 - (1 - r) * (n1 * lambda1 + n2 * lambda2))
    A = (1 + 1 / tf) * z
    def x_equation(x):
        return (A * lambda1 * x + 1 - z * lambda1) ** n1 * (A * lambda2 * x + 1 - z * lambda2) ** n2 - x
    ans1 = fsolve(x_equation, 0, maxfev=500)[0]
    ans2 = fsolve(x_equation, 10, maxfev=500)[0]
    err1 = x_equation(ans1)
    err2 = x_equation(ans2)
    p1 = A * lambda1 + (1 - z * lambda1) * 1 / ans1
    if np.abs(err1) > 1e-5:
        return -1, -1, False
    p2 = A * lambda2 + (1 - z * lambda2) * 1 / ans1
    # print("p1, p2: ", p1, p2)
    return p1, p2, True


def calc_uu_p_formula(nmld: int, mld_lambda: float, nsld: int, sld_lambda: float, tt: float, tf: float)-> Tuple[float, float, bool]:
    """ calc# `p_ps` is being calculated using the `calc_ps_p_fsolve` function, which is used to
    # determine the probability of a partial saturated state in a system. This probability
    # value is then used to make decisions regarding system behavior and resource allocation
    # based on the comparison with other probabilities like `p_as` (probability of
    # all-saturated state).
    ulate p in U-U scenario, return p of each link (both unsaturated)
    Args:
        nmld (int): number of MLDs
        nsld (int): number of SLDs
        mld_lambda: lambda of each link
        tt: tau_T
        tf: tau_F
    Returns:
        pL, pS, uu status or not
    """
    uu = True
    allambda = nmld * mld_lambda + nsld * sld_lambda
    A = (allambda * tf / tt) / (1 - (1 - tf / tt) * allambda)
    B = -(allambda * (1 + tf) / tt) / (1 - (1 - tf / tt) * allambda)
    # pL = exp( np.real(lambertw(B*exp(-A), 0)) + A)
    # pS = exp( np.real(lambertw(B*exp(-A), -1)) + A)
    pL = B / np.real(lambertw(B*exp(-A), 0))
    pS = B / np.real(lambertw(B*exp(-A), -1))
    if not np.isreal(lambertw(B*exp(-A))) or pL >= 1:
        uu = False
    return pL, pS, uu


def calc_ps_p_formula(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    """_summary_

    Args:
        n1 (int): number of mlds
        lambda1 (float): mld's lambda
        n2 (int): number of slds
        lambda2 (float): sld's lambda
        W_1: mld's W
        K_1: mld's K
        W_2: sld's W
        K_2: sld's K
        tt (float): tau_T
        tf (float): tau_F

    Returns:
        p_ps
    """
    # US:
    p_us2 = _calc_ps_p_formula(n2-1, W_2, K_2, n1, lambda1, tt, tf)
    p_us1 = _calc_ps_p_formula(n2, W_2, K_2, n1-1, lambda1, tt, tf)
    # SU:
    p_su1 = _calc_ps_p_formula(n1-1, W_1, K_1, n2, lambda2, tt, tf)
    p_su2 = _calc_ps_p_formula(n1, W_1, K_1, n2-1, lambda2, tt, tf)
    return  p_us1, p_us2, p_su1, p_su2

def calc_ps_p_fsolve(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float)-> float:
    # US:
    p_us1 = calc_PS1(n2, n1-1, W_2, K_2, lambda1, tt, tf)
    p_us2 = calc_PS1(n2-1, n1, W_2, K_2, lambda1, tt, tf)
    # SU:
    p_su1 = calc_PS1(n1-1, n2, W_1, K_1, lambda2, tt, tf)
    p_su2 = calc_PS1(n1, n2-1, W_1, K_1, lambda2, tt, tf)
    return p_us1, p_us2, p_su1, p_su2

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

def calc_ss_p_fsolve(nmld: int, nsld: int, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, bool]:
    """ calculate p in S-S scenario, return p of each link and throughput on each link (both saturated)
    Args:
        M (int): number of links
        nmld (int): number of MLDs
        nsld (List): numbers of SLDs for each link
        kk: kappa of MLDs
        tt: tau_T
        tf: tau_F
    Returns:
        p, is_correct
    """
    ss = True
    pa1, err1 = calc_PA1(nmld, nsld, W_mld, K_mld, W_sld, K_sld)
    pa2, err2 = calc_PA1(nmld, nsld, W_mld, K_mld, W_sld, K_sld)
    if np.abs(err1) > 1e-5:
        ss = False
        return -1, -1, False
    return pa1, ss

def calc_ss_p_formula(nmld: int, nsld: int, W_mld: int, K_mld: int, W_sld: int, K_sld: int)-> Tuple[float, bool]:
    """
    Returns:
        p_as, is_correct
    """
    ss = True
    A = -4 * (nmld / W_mld + nsld / W_sld)
    B = 2 * (nmld / W_mld + nsld / W_sld)
    pL = B / np.real(lambertw(B*exp(-A), 0 ))
    pS = B / np.real(lambertw(B*exp(-A), -1 ))
    if not np.isreal(lambertw(B*exp(-A))):
        ss = False
    return pL , ss
    

# 求解us方程
def _calc_ps_p_formula(n_s: int, W_s: int, K_s: int, n_u: int, lambda_u: float, tt: float, tf: float) -> float:
    def p_func(p, n_s, W_s, K_s, n_u, lambda_u):
        return p - exp(-(n_u * lambda_u) * (1 + tf - tf * p - (tt - tf) * np.log(p) * p) / (tt * p) - 2 * n_s * (2 * p - 1) / (2 * p - 1 + W_s * (p - 2 ** K_s * (1 - p) ** (K_s + 1))))
    ans = -1
    for p in np.arange(0.9999, 0.0001, -0.0001):
        err = np.abs(p_func(p, n_s, W_s, K_s, n_u, lambda_u))
        if err < 1e-3:
            ans = p
            break
    return ans

def calc_PS1(n_s, n_u, W_s, K_s, lambda_u, tt, tf): # nMLD unsaturated, nSLD saturated
    def pf(p, n_s, n_u, W_s, K_s, lambda_u):
        return p - (1 - 2 * (2 * p - 1) / (2 * p - 1 + W_s * (p - 2 ** K_s * (1 - p) ** (K_s + 1))) ) ** n_s *\
                        (1 - lambda_u * (1 + tf - tf * p - (tt - tf) * np.log(p) * p) / (tt * p)) ** n_u
    ans = -1
    for p in np.arange(0.9999, 0.0001, -0.0001):
        err = np.abs(pf(p, n_s, n_u, W_s, K_s, lambda_u))
        if err < 1e-3:
            ans = p
            break
    return ans