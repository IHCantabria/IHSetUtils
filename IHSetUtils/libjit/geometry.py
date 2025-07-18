import numpy as np
from numba import njit
import math

@njit(fastmath=True, cache=True)
def abs_angle_cartesian(relD, batiD):
    n = relD.shape[0]
    waveD = np.empty(n)
    for i in range(n):
        wd = relD[i] + batiD[i]
        if wd > 180.0:
            wd -= 360.0
        elif wd < -180.0:
            wd += 360.0
        waveD[i] = wd
    return waveD

@njit(fastmath=True, cache=True)
def abs_pos(X0, Y0, phi, dn):
    n = dn.shape[0]
    XN = np.empty(n)
    YN = np.empty(n)
    for i in range(n):
        c = math.cos(phi[i])
        s = math.sin(phi[i])
        XN[i] = X0[i] + dn[i] * c
        YN[i] = Y0[i] + dn[i] * s
    return XN, YN

@njit(fastmath=True, cache=True)
def cartesianDir2nauticalDir(cDir):
    n = cDir.shape[0]
    nDir = np.empty(n)
    for i in range(n):
        nd = 90.0 - cDir[i]
        if nd < 0.0:
            nd += 360.0
        nDir[i] = nd
    return nDir

@njit(fastmath=True, cache=True)
def nauticalDir2cartesianDir(nDir):
    n = nDir.shape[0]
    cDir = np.empty(n)
    for i in range(n):
        cd = 90.0 - nDir[i]
        if cd < -180.0:
            cd += 360.0
        cDir[i] = cd
    return cDir

@njit(fastmath=True, cache=True)
def nauticalDir2cartesianDirP(nDir):
    cd = 90.0 - nDir
    if cd < -180.0:
        cd += 360.0
    return cd

@njit(fastmath=True, cache=True)
def pol2cart(rho, phi):
    n = rho.shape[0]
    x = np.empty(n)
    y = np.empty(n)
    for i in range(n):
        x[i] = rho[i] * math.cos(phi[i])
        y[i] = rho[i] * math.sin(phi[i])
    return x, y

@njit(fastmath=True, cache=True)
def rel_angle_cartesian(waveD, batiD):
    n = waveD.shape[0]
    relD = np.empty(n)
    for i in range(n):
        rd = waveD[i] - batiD[i]
        if rd > 180.0:
            rd -= 360.0
        elif rd < -180.0:
            rd += 360.0
        relD[i] = rd
    return relD

@njit(fastmath=True, cache=True)
def rel_angle_cartesianP(waveD, batiD):
    rd = waveD - batiD
    if rd > 180.0:
        rd -= 360.0
    elif rd < -180.0:
        rd += 360.0
    return rd

@njit(fastmath=True, cache=True)
def shore_angle(XN, YN, wave_angle):
    n = XN.shape[0] - 1
    shoreAng = np.empty(n)
    for i in range(n):
        dy = YN[i+1] - YN[i]
        dx = XN[i+1] - XN[i]
        alfa = math.degrees(math.atan2(dy, dx))
        # convert wave to cartesian & rel angle
        cd = nauticalDir2cartesianDirP(wave_angle[i+1])
        relang = rel_angle_cartesianP(cd, alfa)
        ar = abs(relang)
        # branchless selection
        if ar >= 45.0 and ar <= 135.0:
            shoreAng[i] = alfa
        elif ar < 45.0 and i < n-1:
            shoreAng[i] = math.degrees(math.atan2(YN[i+2]-YN[i+1], XN[i+2]-XN[i+1]))
        elif ar > 135.0 and i > 0:
            shoreAng[i] = math.degrees(math.atan2(YN[i]-YN[i-1], XN[i]-XN[i-1]))
        else:
            shoreAng[i] = alfa
    return shoreAng

@njit(fastmath=True, cache=True)
def nauticalDir2cartesianDirL(nDir):
    cd = 90.0 - nDir
    if cd < -180.0:
        cd += 360.0
    return cd

@njit(fastmath=True, cache=True)
def rel_angle_cartesianL(waveD, batiD):
    rd = waveD - batiD
    if rd >= 180.0:
        rd -= 360.0
    elif rd < -180.0:
        rd += 360.0
    return rd

@njit(fastmath=True, cache=True)
def abs_angle_cartesianL(relD, batiD):
    wd = relD + batiD
    if wd > 180.0:
        wd -= 360.0
    elif wd < -180.0:
        wd += 360.0
    return wd

@njit(fastmath=True, cache=True)
def cartesianDir2nauticalDirL(cDir):
    nd = 90.0 - cDir
    if nd < 0.0:
        nd += 360.0
    return nd


# import numpy as np
# from numba import jit

# @jit
# def abs_angle_cartesian(relD, batiD):
#     ###########################################################################    
#     # Absolute angle in cartesian notation, angle between [180,-180], 
#     # 0 is in EAST & positive counterclockwise.
#     # From a relative angle from wave & bathymetry.
#     # The same as rel_angle_cartesian[relD,-1*batiD]
#     # INPUT:
#     # relD:     relative wave angle between wave and bathymetry; 0 is the bathymetry & positive counterclockwise.
#     # batiD:    bathymetry angle (normal to the shoreline) in Cartesian notation.
#     #
#     # OUTPUT:
#     # waveD:    wave angle in Cartesian notation.
#     ###########################################################################    
    
#     waveD = relD + batiD
#     waveD[waveD > 180] = waveD[waveD > 180] - 360
#     waveD[waveD < -180] = waveD[waveD < -180] + 360
    
#     return waveD

# @jit
# def abs_pos(X0, Y0, phi, dn):
#     #####################    
#     # INPUT:
#     #
#     # X0 : x coordinate; origin of the transect
#     # Y0 : y coordinate; origin of the transect
#     # phi : transect orientation in radians
#     # dn : position on the transect
#     #
#     # OUTPUT:
#     # XN : x coordinate
#     # YN : y coordinate
#     #####################

#     XN = X0 + dn * np.cos(phi)
#     YN = Y0 + dn * np.sin(phi)
       
#     return XN, YN

# @jit
# def cartesianDir2nauticalDir(cDir):
#     ###########################################################################    
#     # Cartesian convention with 0 in East & positive counterclockwise TO
#     # Nautical convention with 0 in North & positive clockwise. 
#     ###########################################################################    
    
#     nDir = 90.0 - cDir
#     nDir[nDir < 0] = nDir[nDir < 0] + 360.0
    
#     return nDir

# @jit
# def interp_lon(x, lon, xq, *varargin):
#     # INTERP_LON interpolates a set of longitude angles [in deg]
#     #
#     # Usage: out = interp_lon(x, lon, xq)
#     #
#     # x & lon are vectors of length N.  function evaluates longitude 
#     # (in deg -180..180) at points xq using unwrap & interp1()
#     #
#     # to specify interpolation method used in interp1; use
#     # out = interp_lon(x, lon, xq, METHOD)
#     #
#     # Written by D.G. Long; 27 Nov 2017 
    
#     ulon = np.unwrap(lon * np.pi / 180) * 180 / np.pi
#     if len(varargin) > 0:
#         out = np.interp(x, ulon, xq, varargin[0])
#     else:
#         out = np.interp(x, ulon, xq)
#     out = out % 360
#     out[out > 180] = out[out > 180] - 360
#     return out

# @jit
# def nauticalDir2cartesianDir(nDir):
#     ###########################################################################    
#     # Nautical convention with 0 in North & positive clockwise TO 
#     # Cartesian convention with 0 in East & positive counterclockwise.
#     ###########################################################################    
    

#     cDir = 90.0 - nDir
#     ii = cDir < -180.0

#     cDir[ii] = cDir[ii] + 360.0
    
#     return cDir

# @jit
# def nauticalDir2cartesianDirP(nDir):
#     ###########################################################################    
#     # Nautical convention with 0 in North & positive clockwise TO 
#     # Cartesian convention with 0 in East & positive counterclockwise.
#     ###########################################################################    

#     cDir = 90.0 - nDir
    
#     if cDir < -180.0:
#         cDir = cDir + 360.0
    
#     return cDir

# @jit
# def pol2cart(rho, phi):
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return x, y

# @jit
# def rel_angle_cartesian(waveD, batiD):
#     ###########################################################################    
#     # Relative angle (in degrees) between wave direction & bathymetry with 
#     # angles in cartesian coordinates, angle between [180,-180], 
#     # 0 is in EAST & positive counterclockwise.
#     #
#     # INPUT:
#     # waveD:    wave angle in Cartesian notation.
#     # batiD:    bathymetry angle (normal to the shoreline) in Cartesian notation.
#     #
#     # OUTPUT:
#     # relD:     relative wave angle between wave and bathymetry; 0 is the bathymetry & positive counterclockwise.
#     ###########################################################################    
    
#     relD = waveD - batiD
#     relD[relD > 180.] = relD[relD > 180.] - 360.0
#     relD[relD < -180.] = relD[relD < -180.] + 360.0
    
#     return relD

# @jit    
# def rel_angle_cartesianP(waveD, batiD):
#     ###########################################################################    
#     # Relative angle (in degrees) between wave direction & bathymetry with 
#     # angles in cartesian coordinates, angle between [180,-180], 
#     # 0 is in EAST & positive counterclockwise.
#     #
#     # INPUT:
#     # waveD:    wave angle in Cartesian notation.
#     # batiD:    bathymetry angle (normal to the shoreline) in Cartesian notation.
#     #
#     # OUTPUT:
#     # relD:     relative wave angle between wave and bathymetry; 0 is the bathymetry & positive counterclockwise.
#     ###########################################################################    

#     relD = waveD - batiD
#     if relD > 180:
#         relD = relD - 360.0
#     elif relD < -180.0:
#         relD = relD + 360.0
 
#     return relD

# @jit
# def shore_angle(XN, YN, wave_angle):
#     #####################    
#     # INPUT:
#     #
#     # XN : x coordinate
#     # YN : y coordinate
#     # wave_angle : wave angle()
#     #
#     # OUTPUT:
#     # shoreAng : angle of the shoreline to compute sediment transport
#     #####################
    
#     shoreAng = np.zeros(len(XN) - 1)
#     # method = np.zeros(len(XN) - 1)
    
#     for i in range(len(shoreAng)):
#         alfa = np.arctan2(YN[i + 1] - YN[i], XN[i + 1] - XN[i]) * 180.0 / np.pi
#         relang = rel_angle_cartesianP(nauticalDir2cartesianDirP(wave_angle[i + 1]), alfa)
#         # check the relative angle between the waves & the shoreline orientation [not the normal to the shoreline]
#         if abs(relang) >= 45 and abs(relang) <= 180 - 45:
#             shoreAng[i] = alfa
#         #         method[i] = "CenDiff"
#         elif abs(relang) < 45 and i < len(shoreAng) - 1:
#             try:
#                 shoreAng[i] = np.arctan2(YN[i + 2] - YN[i + 1], XN[i + 2] - XN[i + 1]) * 180.0 / np.pi
#                 #             method[i] = "UpWind"
#             except:
#                 shoreAng[i] = alfa
#                 #             method[i] = "CenDiff"
#         elif abs(relang) > 180 - 45 and i > 1:
#             try:
#                 shoreAng[i] = np.arctan2(YN[i] - YN[i - 1], XN[i] - XN[i - 1]) * 180.0 / np.pi
#                 #             method[i] = "UpWind"
#             except:
#                 shoreAng[i] = alfa
#                 #             method[i] = "CenDiff"
#         else:
#             shoreAng[i] = alfa
#             #             method[i] = "CenDiff"
                
#     return shoreAng


# @jit
# def nauticalDir2cartesianDirL(nDir):
#     ###########################################################################    
#     # Nautical convention with 0 in North & positive clockwise TO 
#     # Cartesian convention with 0 in East & positive counterclockwise.
#     ###########################################################################    
    

#     cDir = 90.0 - nDir
#     if cDir < -180.0:
#         cDir = cDir + 360.0
    
#     return cDir

# @jit
# def rel_angle_cartesianL(waveD, batiD):
#     ###########################################################################    
#     # Relative angle (in degrees) between wave direction & bathymetry with 
#     # angles in cartesian coordinates, angle between [180,-180], 
#     # 0 is in EAST & positive counterclockwise.
#     #
#     # INPUT:
#     # waveD:    wave angle in Cartesian notation.
#     # batiD:    bathymetry angle (normal to the shoreline) in Cartesian notation.
#     #
#     # OUTPUT:
#     # relD:     relative wave angle between wave and bathymetry; 0 is the bathymetry & positive counterclockwise.
#     ###########################################################################    
    
#     relD = waveD - batiD

#     if relD >= 180.0:
#         return relD - 360.0
#     elif relD < -180.0:
#         return relD + 360.0
#     else:
#         return relD

# @jit
# def abs_angle_cartesianL(relD, batiD):
#     ###########################################################################    
#     # Absolute angle in cartesian notation, angle between [180,-180], 
#     # 0 is in EAST & positive counterclockwise.
#     # From a relative angle from wave & bathymetry.
#     # The same as rel_angle_cartesian[relD,-1*batiD]
#     # INPUT:
#     # relD:     relative wave angle between wave and bathymetry; 0 is the bathymetry & positive counterclockwise.
#     # batiD:    bathymetry angle (normal to the shoreline) in Cartesian notation.
#     #
#     # OUTPUT:
#     # waveD:    wave angle in Cartesian notation.
#     ###########################################################################    
    
#     waveD = relD + batiD

#     if waveD > 180.0:
#         return waveD - 360.0
#     elif waveD < -180.0:
#         return waveD + 360.0
    
#     return waveD


# @jit
# def cartesianDir2nauticalDirL(cDir):
#     ###########################################################################    
#     # Cartesian convention with 0 in East & positive counterclockwise TO
#     # Nautical convention with 0 in North & positive clockwise. 
#     ###########################################################################    
    
#     nDir = 90.0 - cDir
#     if nDir < 0:
#         return nDir + 360.0
#     else:
#         return nDir
    
#     return nDir
