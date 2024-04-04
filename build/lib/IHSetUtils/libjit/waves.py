import numpy as np
from numba import jit
from .geometry import rel_angle_cartesian, nauticalDir2cartesianDir, cartesianDir2nauticalDir, abs_angle_cartesian, rel_angle_cartesianL, nauticalDir2cartesianDirL, cartesianDir2nauticalDirL, abs_angle_cartesianL

@jit
def BreakingPropagation(H1, T1, DIR1, h1, bathy_angle, Bcoef):
    ###########################################################################
    # Propagation of waves using linear theory assuming rectilinear & parallel bathymetry
    #
    # INPUT:
    # H1:        wave height.
    # T1:        wave period.
    # DIR1:      wave direction. Nautical convention.
    # h1:        depth of wave conditions.
    # bathy_angle:   bathymetry angle; the normal of the shoreline. Cartesian notation
    # breakType: type of breaking condition. Spectral | monochromatic.
    #
    # OUTPUT:
    # H2:        wave height during breaking. Wave period is assumed invariant due to linear theory
    # DIR2:      wave direction during breaking. Nautical convention.
    # h2:        depth of breaking
    ###########################################################################

    DIRrel = rel_angle_cartesian(nauticalDir2cartesianDir(DIR1), bathy_angle)

    h2l0 = H1 / Bcoef  # initial condition for breaking depth

    H2 = np.zeros_like(H1)
    DIR2 = np.zeros_like(DIR1)
    h2 = np.zeros_like(H1)

    ii = np.greater_equal(h2l0, h1)
    H2[ii] = H1[ii]
    DIR2[ii] = DIR1[ii]
    h2[ii] = h2l0[ii]  # check that the initial depth is deeper than the breaking value

    
    ii = np.less_equal(H1, 0.1)
    H2[ii] = H1[ii]
    DIR2[ii] = DIR1[ii]
    h2[ii] = h2l0[ii]

    idx = (np.abs(DIRrel) <= 90) & (H1 > 0.1) & (h2l0 < h1)
    # idx = np.array(idx)

    if np.sum(idx) > 0:
        h2l = np.zeros_like(h2l0[idx])
        for i, _ in enumerate(h2l0[idx]):
            h2l[i] = find_root_linear_shoal_break(H1[idx][i], T1[idx][i], DIR1[idx][i], h1[idx][i], bathy_angle[idx][i], Bcoef)

        H2l, DIR2l = LinearShoalBreak_ResidualVOL(h2l, H1[idx], T1[idx], DIR1[idx], h1[idx], bathy_angle[idx], Bcoef)
        H2[idx] = H2l
        DIR2[idx] = DIR2l
        h2[idx] = h2l

    return H2, DIR2, h2

@jit
def find_root_linear_shoal_break(H1, T1, DIR1, h1, bathy_angle, Bcoef):
    def f(h2l):
        return LinearShoalBreak_Residual(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef)

    # Bisection method to find the root
    a = 0.01  # Lower bound for the root
    b = H1 / Bcoef  # Upper bound for the root (initial guess based on H1)
    tol = 1e-4  # Tolerance for convergence
    max_iter = 1000  # Maximum number of iterations

    for _ in range(max_iter):
        c = (a + b) / 2  # Midpoint of the interval
        if f(c) == 0 or (b - a) / 2 < tol:
            return c  # Found the root within tolerance
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c

    raise RuntimeError("Failed to converge. Try increasing the maximum number of iterations.")

@jit
def find_root_linear_shoal_break_Brent(H1, T1, DIR1, h1, bathy_angle, Bcoef):

    def f(h2l):
        res = LinearShoalBreak_Residual(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef)
        return res

    # Initial bracketing phase
    a, b = 0.01, H1  # Initial bracket [a, b]

    # Ensure that f(a) and f(b) have different signs
    # if np.sign(f(a)) == np.sign(f(b)):
    #     raise ValueError("Initial bracket [a, b] does not bracket the root.")

    # Main loop using Brent's method
    c, d = a, b
    for _ in range(100):  # Maximum number of iterations
        if np.abs(b - a) < 1e-4:  # Tolerance for convergence
            return (b + a) / 2  # Return the midpoint of the final bracket

        # Interpolation phase
        if f(a) != f(c) and f(b) != f(c):
            s = a * f(b) * f(c) / ((f(a) - f(b)) * (f(a) - f(c))) + \
                b * f(a) * f(c) / ((f(b) - f(a)) * (f(b) - f(c))) + \
                c * f(a) * f(b) / ((f(c) - f(a)) * (f(c) - f(b)))
        else:
            # Bisection phase
            s = b - f(b) * (b - a) / (f(b) - f(a))

        # Update conditions
        m = (a + b) / 2
        tol = 1e-15  # Tolerance for comparisons
        if (np.abs(s - b) < tol or np.abs(b - m) < tol):
            # Convergence criterion: the step is less than the tolerance
            s = m - np.sign(m - a) * tol  # Move slightly inside the bracket
        if np.abs(f(s)) < tol:
            return s  # Found the root within the tolerance

        # Update the bracket
        if f(s) < 0:
            a = s
        else:
            b = s
        if np.abs(f(a)) < np.abs(f(b)):
            a, b = b, a  # Swap a and b if |f(a)| < |f(b)|

    raise RuntimeError("Failed to converge. Try increasing the maximum number of iterations.")

@jit
def GroupCelerity(L, T, h):
    ###########################################################################    
    # CELERITY GROUP
    # L: wave lenght.
    # T: wave period.
    # h: depth of wave conditions.
    ###########################################################################       
    
    c = L / T
    k = 2 * np.pi / L
    N = 1.0 + 2.0 * k * h / np.sinh(2.0 * k * h)
    Cg = c / 2.0 * N
    
    return Cg

@jit
def hunt(T, d):
    ###########################################################################    
    # Wave lenght from Hunt's approximation
    #
    # INPUT:
    # T:     Wave Peak period.
    # d:     Local depth.
    # OUTPUT:
    # L:     Wave length.
    ###########################################################################   
   
    
    g = 9.81

    G = (2 * np.pi / T) ** 2 * (d / g)
    
    F = G + 1.0 / (1 + 0.6522*G + 0.4622*G**2 + 0.0864*G**4 + 0.0675*G**5)

    L = T * (g * d / F) ** 0.5
    
    return L

@jit
def LinearShoal(H1, T1, DIR1, h1, h2, bathy_angle):
    ###########################################################################    
    # Wave shoaling & refraction applying linear theory with parallel; rectilinear bathymetry.
    #    
    # INPUT:
    # H1:        initial wave height.
    # T1:        wave period.
    # DIR1:      initial wave direction. Nautical convention.
    # h1:        initial depth of wave conditions.
    # h2:        final depth of wave conditions.
    # bathy_angle:   bathymetry angle; the normal of the shoreline. Cartesian convention
    #
    # OUTPUT:
    # H2:        wave height during breaking. Wave period is assumed invariant due to linear theory.
    # DIR2:      wave direction during breaking. Nautical convention.
    ###########################################################################

    
    relDir1 = rel_angle_cartesian(nauticalDir2cartesianDir(DIR1), bathy_angle)

    L1 = hunt(T1, h1)
    L2 = hunt(T1, h2)
    CG1 = GroupCelerity(L1, T1, h1)
    CG2 = GroupCelerity(L2, T1, h2)
    relDir2 = Snell_Law(L1, L2, relDir1)
    KS = np.sqrt(CG1 / CG2)
    KR = np.sqrt(np.cos(relDir1 * np.pi / 180) / np.cos(relDir2 * np.pi / 180))
    H2 = H1 * KS * KR
    DIR2 = cartesianDir2nauticalDir(abs_angle_cartesian(relDir2, bathy_angle))
    
    return H2, DIR2

@jit
def LinearShoalBreak_Residual(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef):

    H2l, _ = LinearShoalL(H1, T1, DIR1, h1, h2l, bathy_angle)
    H2comp = h2l * Bcoef
    res = H2l - H2comp

    return res

@jit
def LinearShoalBreak_ResidualVOL(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef):

    H2l, DIR2l = LinearShoal(H1, T1, DIR1, h1, h2l, bathy_angle)
    H2comp = h2l * Bcoef

    return H2l, DIR2l

@jit
def RelDisp(h, T):
    
    g = 9.81
    
    L = hunt(T, h)
    
    Li = hunt(T, h)
    error = 1
    while error > 1E-6:
        L = g * T ** 2 / (2 * np.pi) * np.tanh(2 * np.pi * h / Li)
        error = np.sum(np.sum(np.abs(L - Li)))
        Li = L
    
    C = g * T / (2 * np.pi) * np.tanh(2 * np.pi * h / L)
    
    return L, C

@jit
def RU2_Stockdon2006(slope, hs0, tp):
    ###########################################################################    
    # Run up 2# STOCKDON 2006
    #
    # INPUT:
    # slope:  Beach Slope in swash zone H:V. if slope is V:H all the slope terms multiply
    # hs0:     Significant wave height in deep water.
    # tp:     Peak period.
    #
    # OUTPUT:
    # runup2:   Run-up exceed 2%.
    ###########################################################################
    
    g = 9.81
    L0 = g * tp ** 2 / (2 * np.pi)
    slope = 1.0 / slope  # # V:H
    setup = 0.35 * slope * (hs0 * L0) ** 0.5
    infgr = (hs0 * L0 * (0.563 * slope ** 2 + 0.004)) ** 0.5 / 2
    runup2 = 1.1 * (setup + infgr)  # # eq 19 Stockdon 2006
    return runup2

@jit
def Snell_Law(L1, L2, alpha1):
    ###########################################################################    
    # Wave refraction using snell law.
    #    
    # INPUT:
    # L1:     initial wave length.
    # L1:     final wave length.
    # alpha1: initial wave dir. Cartesian notation.
    #
    # OUTPUT:
    # alpha1: final wave dir. Cartesian notation.
    ###########################################################################    
    alpha = np.arcsin(L2 * np.sin(alpha1 * np.pi / 180) / L1) * 180 / np.pi
    
    return alpha

@jit
def LinearShoalL(H1, T1, DIR1, h1, h2, bathy_angle):
    ###########################################################################    
    # Wave shoaling & refraction applying linear theory with parallel; rectilinear bathymetry.
    #    
    # INPUT:
    # H1:        initial wave height.
    # T1:        wave period.
    # DIR1:      initial wave direction. Nautical convention.
    # h1:        initial depth of wave conditions.
    # h2:        final depth of wave conditions.
    # bathy_angle:   bathymetry angle; the normal of the shoreline. Cartesian convention
    #
    # OUTPUT:
    # H2:        wave height during breaking. Wave period is assumed invariant due to linear theory.
    # DIR2:      wave direction during breaking. Nautical convention.
    ###########################################################################

    
    relDir1 = rel_angle_cartesianL(nauticalDir2cartesianDirL(DIR1), bathy_angle)

    L1 = hunt(T1, h1)
    L2 = hunt(T1, h2)
    CG1 = GroupCelerity(L1, T1, h1)
    CG2 = GroupCelerity(L2, T1, h2)
    relDir2 = Snell_Law(L1, L2, relDir1)
    KS = np.sqrt(CG1 / CG2)
    KR = np.sqrt(np.cos(relDir1 * np.pi / 180) / np.cos(relDir2 * np.pi / 180))
    H2 = H1 * KS * KR
    DIR2 = cartesianDir2nauticalDirL(abs_angle_cartesianL(relDir2, bathy_angle))
    
    return H2, DIR2