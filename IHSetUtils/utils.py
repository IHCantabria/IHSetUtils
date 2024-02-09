import numpy as np
import scipy.io as sio
import xarray as xr
import os

def gow2ncEBSC(gowPath):
    """
    This function converts IH data to netCDF following the EBSC convention.
    """
    wrkDir = os.getcwd()
    # Load the gow file
    gow = sio.loadmat(gowPath)

    # Extract the data
    for key in gow.keys():
        if key.startswith('__'):
            pass
        else:
            data = gow[key]
            break
    
    # Create the xarray
    time = np.squeeze(data['time'][0][0])

    mkY = np.vectorize(lambda t: t.year)
    mkM = np.vectorize(lambda t: t.month)
    mkD = np.vectorize(lambda t: t.day)
    mkH = np.vectorize(lambda t: t.hour)
    mkm = np.vectorize(lambda t: t.minute)
    mks = np.vectorize(lambda t: t.second)

    wav = xr.Dataset(coords={'Y': mkY(time),
                             'M': mkM(time),
                             'D': mkD(time),
                             'h': mkH(time),
                             'm': mkm(time),
                             's': mks(time),
                             'lat': np.squeeze(data['lat'][0][0]),
                             'lon': np.squeeze(data['lon'][0][0])})
    wav['Hs'] = (('Y'), np.squeeze(data['hs'][0][0]))
    wav['Tp'] = (('Y'), 1/np.squeeze(data['fp'][0][0]))
    wav['Dir'] = (('Y'), np.squeeze(data['dir'][0][0]))
    
    wav.to_netcdf(wrkDir+'/data/wav.nc')

    return

def gos2ncEBSC(gosPath):
    """
    This function converts IH data to netCDF following the EBSC convention.
    """
    wrkDir = os.getcwd()
    # Load the gos file
    gos = sio.loadmat(gosPath)

    # Extract the data
    for key in gos.keys():
        if key.startswith('__'):
            pass
        else:
            data = gos[key]
            break
    
    # Create the xarray
    time = np.squeeze(data['time'][0][0])

    mkY = np.vectorize(lambda t: t.year)
    mkM = np.vectorize(lambda t: t.month)
    mkD = np.vectorize(lambda t: t.day)
    mkH = np.vectorize(lambda t: t.hour)
    mkm = np.vectorize(lambda t: t.minute)
    mks = np.vectorize(lambda t: t.second)

    srg = xr.Dataset(coords={'Y': mkY(time),
                             'M': mkM(time),
                             'D': mkD(time),
                             'h': mkH(time),
                             'm': mkm(time),
                             's': mks(time),
                             'lat': np.squeeze(data['lat_zeta'][0][0]),
                             'lon': np.squeeze(data['lon_zeta'][0][0])})
    
    srg['surge'] = (('Y'), np.squeeze(data['zeta'][0][0]))
    
    srg.to_netcdf(wrkDir+'/data/srg.nc')

    return

def got2ncEBSC(gotPath):
    """
    This function converts IH data to netCDF following the EBSC convention.
    """
    wrkDir = os.getcwd()
    # Load the got file
    got = sio.loadmat(gotPath)

    # Extract the data
    for key in got.keys():
        if key.startswith('__'):
            pass
        else:
            data = got[key]
            break
    
    # Create the xarray
    time = np.squeeze(data['time'][0][0])

    mkY = np.vectorize(lambda t: t.year)
    mkM = np.vectorize(lambda t: t.month)
    mkD = np.vectorize(lambda t: t.day)
    mkH = np.vectorize(lambda t: t.hour)
    mkm = np.vectorize(lambda t: t.minute)
    mks = np.vectorize(lambda t: t.second)

    tid = xr.Dataset(coords={'Y': mkY(time),
                             'M': mkM(time),
                             'D': mkD(time),
                             'h': mkH(time),
                             'm': mkm(time),
                             's': mks(time),
                             'lat': np.squeeze(data['lat'][0][0]),
                             'lon': np.squeeze(data['lon'][0][0])})
    tid['tide'] = (('Y'), np.squeeze(data['tide'][0][0]))
    
    tid.to_netcdf(wrkDir+'/data/tid.nc')

    return

# del gost2ncSL():
#     """
#     This function converts IH data to netCDF following the SL convention.
#     """
#     wrkDir = os.getcwd()
#     # Load the gos file
#     gos = sio.loadmat(gosPath)

#     # Extract the data
#     for key in gos.keys():
#         if key.startswith('__'):
#             pass
#         else:
#             data = gos[key]
#             break
    
#     # Create the xarray
#     time = np.squeeze(data['time'][0][0])

#     mkY = np.vectorize(lambda t: t.year)
#     mkM = np.vectorize(lambda t: t.month)
#     mkD = np.vectorize(lambda t: t.day)
#     mkH = np.vectorize(lambda t: t.hour)
#     mkm = np.vectorize(lambda t: t.minute)
#     mks = np.vectorize(lambda t: t.second)

#     srg = xr.Dataset(coords={'Y': mkY(time),
#                              'M': mkM(time),
#                              'D': mkD(time),
#                              'h': mkH(time),
#                              'm': mkm(time),
#                              's': mks(time),
#                              'lat': np.squeeze(data['lat_zeta'][0][0]),
#                              'lon': np.squeeze(data['lon_zeta'][0][0])})
    
#     srg['surge'] = (('Y'), np.squeeze(data['zeta'][0][0]))
    
#     srg.to_netcdf(wrkDir+'/data/srg.nc')

#     return

