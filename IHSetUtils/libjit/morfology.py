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

@njit(nopython=True, fastmath=True, cache=True)
def ALST(Hb, Dirb, hb, bathy_angle, K):
    # Alongshore sediment transport (further optimized)
    n = Hb.shape[0]
    q = np.zeros(n, dtype=np.float64)
    q0 = np.zeros(n, dtype=np.float64)

    # Determine K array only once
    use_scalar_K = (K.shape[0] == 1)
    K0 = K[0] if use_scalar_K else 0.0

    # Constants precomputed
    rho = 1025.0
    rhos_rho = 2650.0 - rho
    inv_rhos_rho = 1.0 / rhos_rho
    sqrt_g = math.sqrt(9.81)
    sin = math.sin
    radians = math.radians

    # Main loop
    for i in range(n):
        H = Hb[i]
        d = hb[i]
        if H > 0.0 and d > 0.0:
            # convert direction & relative angle
            # cd = nauticalDir2cartesianDirP(Dirb[i])
            rel = rel_angle_cartesianP(Dirb[i], bathy_angle[i])
            abs_rel = rel if rel >= 0.0 else -rel
            if abs_rel < 90.0:
                # compute gamma and its sqrt once
                gamma = H / d
                sg = math.sqrt(gamma)
                # count factor
                cnts = rho * sqrt_g * inv_rhos_rho / (16.0 * sg)
                # select K
                Ki = K0 if use_scalar_K else K[i]
                # compute q0 and q
                power = H**2.5
                q0_i = Ki * cnts * power
                q0[i] = q0_i
                q[i] = q0_i * sin(2.0 * radians(rel))
    # apply boundary conditions
    if n > 1:
        q[0] = q[1]
        q[-1] = q[-2]
    return q, q0


# import numpy as np
# from numba import jit
# from .geometry import rel_angle_cartesian, nauticalDir2cartesianDir

# @jit
# def ADEAN(D50):
#     ###########################################################################    
#     # Dean parameter; D50 in meters   
#     ###########################################################################    
#     A = 0.51 * wMOORE(D50) ** 0.44
    
#     return A

# @jit
# def ALST(Hb, Dirb, hb, bathy_angle, K):

#     ###########################################################################    
#     # Alongshore sediment transport
#     #    
#     # INPUT:
#     # Hb:        wave height.
#     # Tb:        wave period.
#     # Dirb:      wave direction. Nautical convention.
#     # hb:        depth of wave conditions.
#     # bathy_angle:   bathymetry angle; the normal of the shoreline in nautical degrees.
#     # method:    alongshore sediment transport formula. Default is CERQ
#     # calp:      calibration parameters
#     # K1:        Komar calibration parameter
#     # K:         SPM calibration parameter
#     #    
#     # OUTPUT:
#     # q:      alongshore sediment transport relative to the bathymetry angle.
#     #
#     # DEPENDENCIAS:
#     # rel_angle_cartesian; rel_dir
#     ###########################################################################

#     if len(K) == 1:
#         K = np.ones_like(Hb) * K

#     rel_dir = rel_angle_cartesian(nauticalDir2cartesianDir(Dirb), bathy_angle)
#     idx_brk = np.abs(rel_dir) < 90
#     q = np.zeros_like(Hb)
#     q0 = np.zeros_like(Hb)

#     rho = 1025 # saltwater mass density SPM
#     rhos = 2650 # sand mass density SPM
#     p = 0.4 # porosity SPM
#     gammab = Hb[idx_brk] / hb[idx_brk]
#     gammab[np.isnan(gammab)] = np.inf
#     cnts = rho * np.sqrt(9.81) / (16. * np.sqrt(gammab) * (rhos - rho) * (1.0 - p))
#     q0[idx_brk] = K[idx_brk] * cnts * (Hb[idx_brk] ** (5. / 2.))
#     q[idx_brk] = q0[idx_brk] * np.sin(2. * np.deg2rad(rel_dir[idx_brk]))

#     q[0] = q[1]
#     q[-1] = q[-2]

#     return q, q0

# @jit
# def BruunRule(hc, D50, Hberm, slr):
#     ###########################################################################    
#     # Bruun Rule
#     # INPUT:
#     # hc:     depth of closure
#     # D50:      Mean sediment grain size (m)
#     # Hberm:    Berm Height [m]
#     # slr:      Expected Sea Level Rise [m]
#     # OUTPUT:
#     # r:        expected progradation/recession [m]
#     #
#     ###########################################################################    
#     Wc = wast(hc, D50)
    
#     r = slr * Wc / (Hberm + hc)
    
#     return r

# @jit
# def depthOfClosure(Hs12, Ts12, type): 
#     '''
#     Closure depth calculation
#     Hs12:     Significant wave height exceed 12 hours in a year.
#     Ts12:     Significant wave period exceed 12 hours in a year.
#     type:     Birkemeier or Hallermeier
#     '''

#     if type == "Birkemeier":
#         dc = 1.75 * Hs12 - 57.9 * (Hs12 ** 2 / (9.81 * Ts12 ** 2))
#     elif type == "Hallermeier":
#         dc = 2.28 * Hs12 - 68.5 * (Hs12 ** 2 / (9.81 * Ts12 ** 2))
        
#     return dc

# @jit
# def Hs12Calc(Hs, Tp):
#     ###########################################################################    
#     # Significant Wave Height exceed 12 hours a year
#     #
#     # INPUT:
#     # Hs:     Significant wave height.
#     # Tp:     Wave Peak period.
#     #
#     # OUTPUT:
#     # Hs12:     Significant wave height exceed 12 hours in a year.
#     # Ts12:     Significant wave period exceed 12 hours in a year.
#     ###########################################################################   
    
#     Hs12calc = np.percentile(Hs, ((365 * 24 - 12) / (365 * 24)) * 100)
#     buscHS12 = (Hs >= (Hs12calc - 0.1)) & (Hs <= (Hs12calc + 0.1))
#     f, xi = np.histogram(Tp[buscHS12], density=True)
#     ii = np.argmax(f)
#     Ts12 = xi[ii]
    
#     return  Hs12calc, Ts12

# @jit
# def wast(hb, D50):
#    ###########################################################################    
#    # Width of the active surf zone
#    # hb:   depth of closure
#    # D50:  mean sediment grain size (m)
#    ###########################################################################    
#     wsf = (hb / ADEAN(D50)) ** (3.0 / 2.0)
#     return wsf

# @jit
# def wMOORE(D50):
#     ###########################################################################    
#     # Fall velocity; D50 in meters. Moore 1982
#     ###########################################################################    
#     if D50 <= 0.1 * 1e-3:
#         ws = 1.1 * 1e6 * D50 ** 2
#     elif D50 > 0.1 * 1e-3 and D50 <= 1.0 * 1e-3:
#         ws = 273.0 * D50 ** 1.1
#     elif D50 > 1 * 1e-3:
#         ws = 4.36 * D50 ** 0.5
#     return ws

# @jit
# def deanSlope(depth, D50):
#     ###########################################################################    
#     # Slope for a Dean profile; D50 and depth in meters
#     ###########################################################################    
#     A = ADEAN(D50)
#     x = wast(depth, D50)
#     return 2 * A / (3 * (x) ** (1 / 3))
