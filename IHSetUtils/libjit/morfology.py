import numpy as np
from numba import njit
import math

# Dependencias de wave_utils
from .geometry import (
    rel_angle_cartesianP
)

@njit(nopython=True, fastmath=True, cache=True)
def wMOORE(D50):
    # Fall velocity; D50 in meters. Moore 1982
    if D50 <= 0.1e-3:
        return 1.1e6 * D50 * D50
    elif D50 <= 1e-3:
        return 273.0 * D50 ** 1.1
    else:
        return 4.36 * math.sqrt(D50)

@njit(nopython=True, fastmath=True, cache=True)
def ADEAN(D50):
    # Dean parameter; D50 in meters
    return 0.51 * wMOORE(D50) ** 0.44

@njit(nopython=True, fastmath=True, cache=True)
def A_Dean_Dalrymple(D50):
    # Dean parameter; D50 in meters
    return 2.25 * (wMOORE(D50)**2/9.81) ** (1/3)

@njit(nopython=True, fastmath=True, cache=True)
def wast(hb, D50):
    # Width of the active surf zone
    # hb and D50 scalars
    return (hb / ADEAN(D50)) ** 1.5

@njit(nopython=True, fastmath=True, cache=True)
def deanSlope(depth, D50):
    # Slope for a Dean profile; D50 and depth in meters
    A = ADEAN(D50)
    x = wast(depth, D50)
    return 2.0 * A / (3.0 * x ** (1.0/3.0))

@njit(nopython=True, fastmath=True, cache=True)
def BruunRule(hc, D50, Hberm, slr):
    # Expected progradation/recession
    Wc = wast(hc, D50)
    return slr * Wc / (Hberm + hc)

@njit(nopython=True, fastmath=True, cache=True)
def depthOfClosure(Hs12, Ts12, typeflag):
    # typeflag: 0 -> Birkemeier, 1 -> Hallermeier
    if typeflag == 0:
        return 1.75 * Hs12 - 57.9 * (Hs12 * Hs12 / (9.81 * Ts12 * Ts12))
    else:
        return 2.28 * Hs12 - 68.5 * (Hs12 * Hs12 / (9.81 * Ts12 * Ts12))

@njit(nopython=True, fastmath=True, cache=True)
def Hs12Calc(Hs, Tp):
    # Significant wave height exceed 12 hours a year
    n = Hs.shape[0]
    # Compute percentile index
    idx = int(((365*24 - 12) / (365*24)) * n)
    # Sort Hs
    sortedHs = np.sort(Hs)
    Hs12 = sortedHs[idx]
    # Compute mean Tp for Hs near Hs12
    sum_tp = 0.0
    count = 0
    lower = Hs12 - 0.1
    upper = Hs12 + 0.1
    for i in range(n):
        if Hs[i] >= lower and Hs[i] <= upper:
            sum_tp += Tp[i]
            count += 1
    if count > 0:
        return Hs12, sum_tp / count
    else:
        return Hs12, Tp[0]

@njit(fastmath=True, cache=True)
def ALST(Hb, Tp, Dirb, hb, bathy_angle, K, mb, D50, formula):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    sin = math.sin
    cos = math.cos
    sqrt = math.sqrt
    radians = math.radians
    abs = math.fabs

    if formula == 1:  # SPM
        # Main loop
        rho = 1025.0
        rhos_rho = 2650.0 - rho
        lbda = 0.4
        cnts = rho / (16.0 * lbda * (rhos_rho))
        for i in range(n):
            H = Hb[i]
            d = hb[i]
            if H > 0.0 and d > 0.0:
                # convert direction & relative angle
                # cd = nauticalDir2cartesianDirP(Dirb[i])
                rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
                abs_rel = rel if rel >= 0.0 else -rel
                if abs_rel < 90.0:
                    Ki = K0 if use_scalar_K else K[i]
                    sqrt_gd = sqrt(9.81*d)
                    times = cnts * sqrt_gd
                    powerH = H ** 2
                    q0_i = Ki * times * powerH
                    q0[i] = q0_i
                    q[i] = q0[i]*sin(2*radians(rel))

    elif formula == 2:  # KOMAR
        # Main loop
        rho = 1025.0
        rhos_rho = 2650.0 - rho
        powerg = 9.81 ** 1.5
        cnts = rho * powerg
        for i in range(n):
            H = Hb[i]
            d = hb[i]
            if H > 0.0 and d > 0.0:
                # convert direction & relative angle
                # cd = nauticalDir2cartesianDirP(Dirb[i])
                rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
                abs_rel = rel if rel >= 0.0 else -rel
                if abs_rel < 90.0:
                    Ki = K0 if use_scalar_K else K[i]
                    # compute q0 and q
                    power = H**2.5
                    q0_i = Ki * cnts * power
                    q0[i] = q0_i
                    q[i] = q0_i * sin(radians(rel)) * cos(radians(rel))

    elif formula == 3:  # Kamphuis
        # Main loop
        powerD50 = D50 ** (-0.25)
        powermb = mb ** (0.75)
        rho = 1025.0
        rhos_rho = 2650.0 - rho
        inv_sqrt_rhos_rho = 1.0 / (rhos_rho * 9.81 * (1.0 - 0.4))
        cnts = powerD50 * powermb * inv_sqrt_rhos_rho
        for i in range(n):
            H = Hb[i]
            d = hb[i]
            T = Tp[i]
            if H > 0.0 and d > 0.0:
                Ki = K0 if use_scalar_K else K[i]
                # convert direction & relative angle
                # cd = nauticalDir2cartesianDirP(Dirb[i])
                rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
                abs_rel = rel if rel >= 0.0 else -rel
                if rel < 90.0:
                    # compute gamma and its sqrt once
                    powerHb = H ** 2
                    powerT = T ** 1.5
                    q0_i = cnts * powerHb * powerT * Ki
                    q0[i] = q0_i
                    if rel >= 0.0:
                        q[i] = q0_i * sin(2* radians(rel)) ** 0.6
                    else:
                        q[i] = -q0_i * sin(2* radians(abs(rel))) ** 0.6

    elif formula == 4:  # Van Rijn
        # Main loop
        sqrt_g = sqrt(9.81)
        powerD50 = D50 ** (-0.6)
        powermb = mb ** (0.4)
        cnts = 0.00018* sqrt_g / (1.0 - 0.4) * powerD50 * powermb
        for i in range(n):
            H = Hb[i]
            d = hb[i]
            if H > 0.0 and d > 0.0:
                Ki = K0 if use_scalar_K else K[i]
                # convert direction & relative angle
                rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
                abs_rel = rel if rel >= 0.0 else -rel
                if abs_rel < 90.0:
                    # compute q0 and q
                    powerH = H ** 3.1
                    q0_i = Ki * cnts * powerH 
                    q0[i] = q0_i
                    q[i] = q0[i] * sin(2 * radians(rel))

    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0

@njit(fastmath=True, cache=True)
def CERQ_ALST(Hb, Dirb, hb, bathy_angle, K):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    sin = math.sin
    sqrt = math.sqrt
    radians = math.radians

    rho = 1025.0
    rhos_rho = 2650.0 - rho
    lbda = 0.4
    cnts = rho / (16.0 * lbda * (rhos_rho))
    for i in range(n):
        H = Hb[i]
        d = hb[i]
        if H > 0.0 and d > 0.0:
            # convert direction & relative angle
            # cd = nauticalDir2cartesianDirP(Dirb[i])
            rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
            abs_rel = rel if rel >= 0.0 else -rel
            if abs_rel < 90.0:
                Ki = K0 if use_scalar_K else K[i]
                sqrt_gd = sqrt(9.81*d)
                times = cnts * sqrt_gd
                powerH = H ** 2
                q0_i = Ki * times * powerH
                q0[i] = q0_i
                q[i] = q0[i]*sin(2*radians(rel))
    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0

@njit(fastmath=True, cache=True)
def Komar_ALST(Hb, Dirb, hb, bathy_angle, K):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    sin = math.sin
    cos = math.cos
    radians = math.radians

    # Main loop
    rho = 1025.0
    rhos_rho = 2650.0 - rho
    powerg = 9.81 ** 1.5
    cnts = rho * powerg
    for i in range(n):
        H = Hb[i]
        d = hb[i]
        if H > 0.0 and d > 0.0:
            # convert direction & relative angle
            # cd = nauticalDir2cartesianDirP(Dirb[i])
            rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
            abs_rel = rel if rel >= 0.0 else -rel
            if abs_rel < 90.0:
                Ki = K0 if use_scalar_K else K[i]
                # compute q0 and q
                power = H**2.5
                q0_i = Ki * cnts * power
                q0[i] = q0_i
                q[i] = q0_i * sin(radians(rel)) * cos(radians(rel))

    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0


@njit(fastmath=True, cache=True)
def Kamphuis_ALST(Hb, Tp, Dirb, hb, bathy_angle, K, mb, D50):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    sin = math.sin
    radians = math.radians
    abs = math.fabs

    # Main loop
    powerD50 = D50 ** (-0.25)
    powermb = mb ** (0.75)
    rho = 1025.0
    rhos_rho = 2650.0 - rho
    inv_sqrt_rhos_rho = 1.0 / (rhos_rho * 9.81 * (1.0 - 0.4))
    cnts = powerD50 * powermb * inv_sqrt_rhos_rho
    for i in range(n):
        H = Hb[i]
        d = hb[i]
        T = Tp[i]
        if H > 0.0 and d > 0.0:
            Ki = K0 if use_scalar_K else K[i]
            # convert direction & relative angle
            # cd = nauticalDir2cartesianDirP(Dirb[i])
            rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
            abs_rel = rel if rel >= 0.0 else -rel
            if rel < 90.0:
                # compute gamma and its sqrt once
                powerHb = H ** 2
                powerT = T ** 1.5
                q0_i = cnts * powerHb * powerT * Ki
                q0[i] = q0_i
                if rel >= 0.0:
                    q[i] = q0_i * sin(2* radians(rel)) ** 0.6
                else:
                    q[i] = -q0_i * sin(2* radians(abs(rel))) ** 0.6

    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0

@njit(fastmath=True, cache=True)
def VanRijn_ALST(Hb, Dirb, hb, bathy_angle, K, mb, D50):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    sin = math.sin
    sqrt = math.sqrt
    radians = math.radians

    # Main loop
    sqrt_g = sqrt(9.81)
    powerD50 = D50 ** (-0.6)
    powermb = mb ** (0.4)
    cnts = 0.00018* sqrt_g / (1.0 - 0.4) * powerD50 * powermb
    for i in range(n):
        H = Hb[i]
        d = hb[i]
        if H > 0.0 and d > 0.0:
            Ki = K0 if use_scalar_K else K[i]
            # convert direction & relative angle
            rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
            abs_rel = rel if rel >= 0.0 else -rel
            if abs_rel < 90.0:
                # compute q0 and q
                powerH = H ** 3.1
                q0_i = Ki * cnts * powerH 
                q0[i] = q0_i
                q[i] = q0[i] * sin(2 * radians(rel))

    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0