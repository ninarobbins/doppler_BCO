import numpy as np


def compute_reflectivity_MBR2(SPC_co, HSDV, RadarConst, range_val, COFA, NPW1):
    """
    Calculate total reflectivity Z using the formula in MBRS RadarConst5 metadata.
    2.21 is the correction included by Lutz (MBRS to MBR2 correction).

    Parameters:
    - SPC_co: Array of spectral power components (SPC values).
    - HSD_co: Threshold for SPC values to be included in the sum.
    - RadarConst: Radar constant, typically at 5 km.
    - range_val: Range in meters for which reflectivity is calculated.
    - COFA: Signal-to-noise ratio correction factor.
    - NPW1: Normalization factor (defaults to 1 if not provided).

    Returns:
    - Reflectivity Z.
    """
    
    SPCcoi = np.ma.masked_less_equal(SPC_co.transpose(), HSDV)
    SPCcoi = (SPCcoi-HSDV).transpose()
    Z = np.nansum(SPCcoi, axis = 1)*RadarConst*(range_val/5e3)**2*COFA/NPW1/2.21

    return 10*np.log10(Z)


def compute_reflectivity_MBRS(SPC_co, HSDV, RadarConst, range_val, COFA, NPW1):
    """
    Calculate total reflectivity Z using the formula in MBRS RadarConst5 metadata.
    2.21 is the correction included by Lutz (MBRS to MBR2 correction).

    Parameters:
    - SPC_co: Array of spectral power components (SPC values).
    - HSD_co: Threshold for SPC values to be included in the sum.
    - RadarConst: Radar constant, typically at 5 km.
    - range_val: Range in meters for which reflectivity is calculated.
    - COFA: Signal-to-noise ratio correction factor.
    - NPW1: Normalization factor (defaults to 1 if not provided).

    Returns:
    - Reflectivity Z.
    """
    
    SPCcoi = np.ma.masked_less_equal(SPC_co.transpose(), HSDV)
    SPCcoi = (SPCcoi-HSDV).transpose()
    Z = np.nansum(SPCcoi, axis = 1)*RadarConst*(range_val/5e3)**2*COFA/NPW1

    return 10*np.log10(Z)
