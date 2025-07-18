import numpy as np
from .geometry import rel_angle_cartesian, nauticalDir2cartesianDir

def ADEAN(D50):
    ###########################################################################    
    # Dean parameter; D50 in meters   
    ###########################################################################    
    A = 0.51 * wMOORE(D50) ** 0.44
    
    return A

def ALST(Hb, Dirb, hb, bathy_angle, K):

    ###########################################################################    
    # Alongshore sediment transport
    #    
    # INPUT:
    # Hb:        wave height.
    # Tb:        wave period.
    # Dirb:      wave direction. Nautical convention.
    # hb:        depth of wave conditions.
    # bathy_angle:   bathymetry angle; the normal of the shoreline in nautical degrees.
    # method:    alongshore sediment transport formula. Default is CERQ
    # calp:      calibration parameters
    # K1:        Komar calibration parameter
    # K:         SPM calibration parameter
    #    
    # OUTPUT:
    # q:      alongshore sediment transport relative to the bathymetry angle.
    #
    # DEPENDENCIAS:
    # rel_angle_cartesian; rel_dir
    ###########################################################################

    if len(K) == 1:
        K = np.ones_like(Hb) * K

    rel_dir = rel_angle_cartesian(nauticalDir2cartesianDir(Dirb), bathy_angle)
    idx_brk = np.abs(rel_dir) < 90
    q = np.zeros_like(Hb)
    q0 = np.zeros_like(Hb)

    rho = 1025 # saltwater mass density SPM
    rhos = 2650 # sand mass density SPM
    p = 0.4 # porosity SPM
    gammab = Hb[idx_brk] / hb[idx_brk]
    gammab[np.isnan(gammab)] = np.inf
    cnts = rho * np.sqrt(9.81) / (16. * np.sqrt(gammab) * (rhos - rho) * (1.0 - p))
    q0[idx_brk] = K[idx_brk] * cnts * (Hb[idx_brk] ** (5. / 2.))
    q[idx_brk] = q0[idx_brk] * np.sin(2. * np.deg2rad(rel_dir[idx_brk]))

    q[0] = q[1]
    q[-1] = q[-2]

    return q, q0

def BruunRule(hc, D50, Hberm, slr):
    ###########################################################################    
    # Bruun Rule
    # INPUT:
    # hc:     depth of closure
    # D50:      Mean sediment grain size (m)
    # Hberm:    Berm Height [m]
    # slr:      Expected Sea Level Rise [m]
    # OUTPUT:
    # r:        expected progradation/recession [m]
    #
    ###########################################################################    
    Wc = wast(hc, D50)
    
    r = slr * Wc / (Hberm + hc)
    
    return r

def depthOfClosure(Hs12, Ts12, type="Birkemeier"): 
    '''
    Closure depth calculation. Birkemeier[1985](default) or Hallermeier[1978]
    Hs12:     Significant wave height exceed 12 hours in a year.
    Ts12:     Significant wave period exceed 12 hours in a year.
    '''

    if type == "Birkemeier":
        dc = 1.75 * Hs12 - 57.9 * (Hs12 ** 2 / (9.81 * Ts12 ** 2))
    elif type == "Hallermeier":
        dc = 2.28 * Hs12 - 68.5 * (Hs12 ** 2 / (9.81 * Ts12 ** 2))
        
    return dc

def Hs12Calc(Hs, Tp):
    ###########################################################################    
    # Significant Wave Height exceed 12 hours a year
    #
    # INPUT:
    # Hs:     Significant wave height.
    # Tp:     Wave Peak period.
    #
    # OUTPUT:
    # Hs12:     Significant wave height exceed 12 hours in a year.
    # Ts12:     Significant wave period exceed 12 hours in a year.
    ###########################################################################   
    
    Hs12calc = np.percentile(Hs, ((365 * 24 - 12) / (365 * 24)) * 100)
    buscHS12 = (Hs >= (Hs12calc - 0.1)) & (Hs <= (Hs12calc + 0.1))
    f, xi = np.histogram(Tp[buscHS12], density=True)
    ii = np.argmax(f)
    Ts12 = xi[ii]
    
    return  Hs12calc, Ts12

def wast(hb, D50):
   ###########################################################################    
   # Width of the active surf zone
   # hb:   depth of closure
   # D50:  mean sediment grain size (m)
   ###########################################################################    
    wsf = (hb / ADEAN(D50)) ** (3.0 / 2.0)
    return wsf

def wMOORE(D50):
    ###########################################################################    
    # Fall velocity; D50 in meters. Moore 1982
    ###########################################################################    
    if D50 <= 0.1 * 1e-3:
        ws = 1.1 * 1e6 * D50 ** 2
    elif D50 > 0.1 * 1e-3 and D50 <= 1.0 * 1e-3:
        ws = 273.0 * D50 ** 1.1
    elif D50 > 1 * 1e-3:
        ws = 4.36 * D50 ** 0.5
    return ws

def deanSlope(depth, D50):
    ###########################################################################    
    # Slope for a Dean profile; D50 and depth in meters
    ###########################################################################    
    A = ADEAN(D50)
    x = wast(depth, D50)
    return 2 * A / (3 * (x) ** (1 / 3))
