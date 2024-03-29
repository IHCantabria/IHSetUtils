import numpy as np
import scipy.optimize as optimize
from .geometry import rel_angle_cartesian, nauticalDir2cartesianDir, cartesianDir2nauticalDir, abs_angle_cartesian

def BreakingPropagation(H1, T1, DIR1, h1, bathy_angle, breakType):
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

    if breakType == "mono":
        Bcoef = 0.78
    elif breakType == "spectral":
        Bcoef = 0.45

    DIRrel = rel_angle_cartesian(nauticalDir2cartesianDir(DIR1), bathy_angle)

    h2l0 = H1 / Bcoef  # initial condition for breaking depth

    H2 = np.zeros_like(H1)
    DIR2 = np.zeros_like(DIR1)
    h2 = np.zeros_like(H1)

    
    H2[h2l0 >= h1] = H1[h2l0 >= h1]
    DIR2[h2l0 >= h1] = DIR1[h2l0 >= h1]
    h2[h2l0 >= h1] = h2l0[h2l0 >= h1]  # check that the initial depth is deeper than the breaking value

    H2[H1 <= 0.1] = H1[H1 <= 0.1]
    DIR2[H1 <= 0.1] = DIR1[H1 <= 0.1]
    h2[H1 <= 0.1] = h2l0[H1 <= 0.1]

    idx = (np.abs(DIRrel) <= 90) & (H1 > 0.1) & (h2l0 < h1)
    idx = np.array(idx)

    if np.sum(idx) > 0:

        def myFun(x):
            return LinearShoalBreak_Residual(x, H1[idx], T1[idx], DIR1[idx], h1[idx], bathy_angle[idx], Bcoef)

        h2l = optimize.newton_krylov(myFun, h2l0[idx], method="minres")

        H2l, DIR2l = LinearShoalBreak_ResidualVOL(h2l, H1[idx], T1[idx], DIR1[idx], h1[idx], bathy_angle[idx], Bcoef)
        H2[idx] = H2l
        DIR2[idx] = DIR2l
        h2[idx] = h2l

    return H2, DIR2, h2

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
    
    p = np.poly1d([0.0675, 0.0864, 0.4622, 0.6522, 1.0])
    
    F = G + 1.0 / p(G)

    L = T * (g * d / F) ** 0.5
    
    return L

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

def LinearShoalBreak_Residual(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef):

    H2l, _ = LinearShoal(H1, T1, DIR1, h1, h2l, bathy_angle)
    H2comp = h2l * Bcoef
    res = H2l - H2comp

    return res

def LinearShoalBreak_ResidualVOL(h2l, H1, T1, DIR1, h1, bathy_angle, Bcoef):

    H2l, DIR2l = LinearShoal(H1, T1, DIR1, h1, h2l, bathy_angle)
    H2comp = h2l * Bcoef
    res = H2l - H2comp

    return H2l, DIR2l

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
