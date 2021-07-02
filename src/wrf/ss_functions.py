"""
various routines and subroutines for calculating predicted 
superaturation from WRF simulation data
"""
import numpy as np
import re

##
## physical constants
##
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

##
## least squares regression from wrf data 
## 
LSR_INT = 0.04164662760767679
LSR_SLOPE = 0.8253679031561234

##
## methods to get ss_qss 
##
def get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, full_ss, incl_rain, incl_vent):

    meanr = get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)

    w = met_vars['w'][...]

    if full_ss:
        A = met_vars['A'][...]
        B = met_vars['B'][...]
    else:
        temp = met_vars['temp'][...]
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        B = D

    ss_qss = A*w/(4*np.pi*B*meanr*nconc)*100. #as a percentage

    return ss_qss

def get_ss_qss_components(met_vars, dsdsum_vars, cutoff_bins, full_ss, incl_rain, incl_vent):

    meanr = get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)

    w = met_vars['w'][...]

    if full_ss:
        A = met_vars['A'][...]
        B = met_vars['B'][...]
    else:
        temp = met_vars['temp'][...]
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        B = D

    return A, B, meanr, nconc 

def get_ss_pred(ss_qss):

    ss_pred = LSR_INT + LSR_SLOPE*ss_qss

    return ss_pred

##
## methods to get meanr and nconc 
##
def get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent):

    suffixes = ['mid']

    if not cutoff_bins:
        suffixes.append('lo')
    if incl_rain:
        suffixes.append('hi')

    prefix = ''
    
    if incl_vent:
        prefix += 'f'

    var_names = [prefix + 'rn_sum_' + suffix for suffix in suffixes]

    nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent) 
    meanr = np.zeros(np.shape(dsdsum_vars['nconc_sum_lo'][...])) 

    for var_name in var_names:
        meanr += dsdsum_vars[var_name][...]

    return meanr/nconc

def get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent):

    suffixes = ['mid']

    if not cutoff_bins:
        suffixes.append('lo')
    if incl_rain:
        suffixes.append('hi')

    nconc = np.zeros(np.shape(dsdsum_vars['nconc_sum_lo'][...])) 

    var_names = ['nconc_sum_' + suffix for suffix in suffixes]

    for var_name in var_names:
        nconc += dsdsum_vars[var_name][...]

    return nconc

###
### method to get LWC from cloud droplets only - still have incl_rain
### param cause I'm too lazy to go around and remove it everywhere
### 
def get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent):

    suffixes = ['mid']

    if not cutoff_bins:
        suffixes.append('lo')

    prefix = ''
    
    if incl_vent:
        prefix += 'f'

    r3n = np.zeros(np.shape(dsdsum_vars['nconc_sum_lo'][...])) 

    var_names = [prefix + 'r3n_sum_' + suffix for suffix in suffixes]

    for var_name in var_names:
        r3n += dsdsum_vars[var_name][...]

    rho_air = met_vars['rho_air'][...]
    lwc = 4./3.*np.pi*rho_w*r3n/rho_air

    return lwc

def linregress(x, y=None):
    """
    ~~copy pasta from scipy so I don't have to import the whole damn module~~
    Calculate a regression line
    This computes a least-squares regression for two sets of measurements.
    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.
    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    r-value : float
        correlation coefficient
    stderr : float
        Standard error of the estimate
    """
    TINY = 1.0e-20
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) \
            or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    n = len(x)
    xmean = np.mean(x,None)
    ymean = np.mean(y,None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm*ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if (r > 1.0):
            r = 1.0
        elif (r < -1.0):
            r = -1.0

    df = n-2
    t = r*np.sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)))
    slope = r_num / ssxm
    intercept = ymean - slope*xmean
    sterrest = np.sqrt((1-r*r)*ssym / ssxm / df)
    return slope, intercept, r, sterrest
