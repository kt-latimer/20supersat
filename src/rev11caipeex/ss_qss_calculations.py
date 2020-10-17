"""
various routines and subroutines for calculating quasi-steady-state
superaturation from CAIPEEX campaign data
*random and non-comprehensive notes*
-abandoning 'meanfr' notation and just putting boolean incl_vent arg
-option to calculate full ss_qss is also now a boolean arg
"""
import numpy as np
import re

from revcaipeex import CDP_bins

##
## center radii of bins
##
CDP_bin_radii = (CDP_bins['upper'] + CDP_bins['lower'])/4.

##
## various series expansion coeffs - comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130
N_Re_regime2_coeffs = [-0.318657e1, 0.992696, -0.153193e-2, \
                        -0.987059e-3, -0.578878e-3, 0.855176e-4, \
                        -0.327815e-5] #417
N_Re_regime3_coeffs = [-0.500015e1, 0.523778e1, -0.204914e1, \
                        0.475294, -0.542819e-1, 0.238449e-2] #418

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
## methods to get ss_qss 
##
def get_ss_vs_t(met_dict, cpd_dict, \
                cutoff_bins, full_ss, incl_rain, incl_vent):

    meanr = get_meanr_vs_t(met_dict, cpd_dict, \
                    cutoff_bins, incl_rain, incl_vent)
    nconc = get_nconc_vs_t(met_dict, cpd_dict, \
                    cutoff_bins, incl_rain, incl_vent)

    temp = met_dict['data']['temp']
    w = met_dict['data']['w']

    if full_ss:
        pres = met_dict['data']['pres']
        rho_air = pres/(R_a*temp) 
        e_s = get_sat_vap_pres(temp)
        F_d = rho_w*R_v*temp/(D*e_s) 
        F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp*(F_d + F_k)
        B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_air*temp**2.)) 
    else:
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        B = D

    ss_qss = A*w/(4*np.pi*B*meanr*nconc)*100. #as a percentage
    return ss_qss

##
## methods to get meanr and nconc 
##
def get_meanr_vs_t(met_dict, cpd_dict, cutoff_bins, \
                            incl_rain, incl_vent):

    nconc = get_nconc_vs_t(met_dict, cpd_dict, cutoff_bins, \
                                    incl_rain, incl_vent)

    meanr = np.zeros(np.shape(met_dict['data']['time']))

    for var_name in cpd_dict['data'].keys():
        meanr += get_meanr_contribution_from_cpd_var(var_name, \
                    met_dict, cpd_dict, cutoff_bins, incl_rain, incl_vent)

    return meanr/nconc

def get_nconc_vs_t(met_dict, cpd_dict, cutoff_bins, \
                            incl_rain, incl_vent):

    nconc = np.zeros(np.shape(met_dict['data']['time']))

    for var_name in cpd_dict['data'].keys():
        nconc += get_nconc_contribution_from_cpd_var(var_name, \
                    met_dict, cpd_dict, cutoff_bins, incl_rain, incl_vent)

    return nconc

##
## methods to get meanr and nconc contributions from cpd data. 
## note: meanr contribution is weighted by nconc for that bin
##
def get_meanr_contribution_from_cpd_var(var_name, met_dict, cpd_dict, \
                                        cutoff_bins, incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(met_dict['data']['time']))

    if 'diam' in var_name:
        return zero_arr
    
    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except IndexError: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 1 
    r = CDP_bin_radii[bin_ind]

    pres = met_dict['data']['pres']
    temp = met_dict['data']['temp']

    rho_air = pres/(R_a*temp)
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air**2./(eta**4.*g*rho_w) #pr&kl p 418

    u_term = get_u_term(r, eta, N_Be_div_r3, N_Bo_div_r2, \
                            N_P, pres, rho_air, temp)
    N_Re = 2*rho_air*r*u_term/eta
    f = get_ventilation_coefficient(N_Re, incl_vent)

    nconc_contribution_from_var = get_nconc_contribution_from_cpd_var( \
        var_name, met_dict, cpd_dict, cutoff_bins, \
        incl_rain, incl_vent)

    mean_r_contribution_from_var = nconc_contribution_from_var*r*f

    return mean_r_contribution_from_var

def get_nconc_contribution_from_cpd_var(var_name, met_dict, cpd_dict, \
                        cutoff_bins, incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(met_dict['data']['time']))
    
    is_bin_var = check_if_bin_var(var_name)
    if not is_bin_var:
        return zero_arr 

    has_correct_lower_bin_cutoff = \
        has_correct_lower_bin_cutoff_cpd(var_name, cutoff_bins)
    if not has_correct_lower_bin_cutoff:
        return zero_arr 

    has_correct_upper_bin_cutoff = \
        has_correct_upper_bin_cutoff_cpd(var_name, incl_rain)
    if not has_correct_upper_bin_cutoff:
        return zero_arr 

    nconc_contribution_from_var = cpd_dict['data'][var_name]

    return nconc_contribution_from_var
    
##
## methods to get ventilation factor
##
def get_u_term(r, eta, N_Be_div_r3, N_Bo_div_r2, N_P, pres, rho_air, temp):
    """
    get terminal velocity for cloud / rain droplet of radius r given ambient
    temperature and pressure (from pruppacher and klett pp 415-419)
    """
    if r <= 10.e-6:
        lam = 6.6e-8*(10132.5/pres)*(temp/293.15)
        u_term = (1 + 1.26*lam/r)*(2*r**2.*g*rho_w/9*eta)
    elif r <= 535.e-6:
        N_Be = N_Be_div_r3*r**3.
        X = np.log(N_Be)
        N_Re = np.exp(sum([N_Re_regime2_coeffs[i]*X**i for i in \
                        range(len(N_Re_regime2_coeffs))]))
        u_term = eta*N_Re/(2*rho_air*r)
    else:
        N_Bo = N_Bo_div_r2*r**2.
        X = np.log(16./3.*N_Bo*N_P**(1./6.))
        N_Re = N_P**(1./6.)*np.exp(sum([N_Re_regime3_coeffs[i]*X**i for i in \
                                    range(len(N_Re_regime3_coeffs))]))
        u_term = eta*N_Re/(2*rho_air*r)
    return u_term

def get_dyn_visc(temp):
    """
    get dynamic viscocity as a function of temperature (from pruppacher and
    klett p 417)
    """
    eta = np.piecewise(temp, [temp < 273, temp >= 273], \
                        [lambda temp: (1.718 + 0.0049*(temp - 273) \
                                    - 1.2e-5*(temp - 273)**2.)*1.e-5, \
                        lambda temp: (1.718 + 0.0049*(temp - 273))*1.e-5])
    return eta

def get_ventilation_coefficient(N_Re, incl_vent):
    """
    get ventilation coefficient (from pruppacher and klett p 541)
    """

    if not incl_vent:
        return 1.

    f = np.piecewise(N_Re, [N_Re < 2.46, N_Re >= 2.46], \
                    [lambda N_Re: 1. + 0.086*N_Re, \
                    lambda N_Re: 0.78 + 0.27*N_Re**0.5])
    return f

##
## methods to determine if variable is appropriate for various specified
## boolean arguments
##
def check_if_bin_var(var_name):

    if 'nconc' in var_name and 'cdp' not in var_name:
        return True
    else:
        return False

def has_correct_lower_bin_cutoff_cpd(var_name, cutoff_bins):

    if not cutoff_bins:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 1 
    lower_bin_radius = CDP_bins['lower'][bin_ind]
    
    if lower_bin_radius >= 5.e-6:
        return True
    else:
        return False

def has_correct_upper_bin_cutoff_cpd(var_name, incl_rain):

    if incl_rain:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 1 
    upper_bin_radius = CDP_bins['upper'][bin_ind]
    
    if upper_bin_radius <= 50.e-6: 
        return True
    else:
        return False

##
## method to get saturation vapor pressure
##
def get_sat_vap_pres(temp):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
    return e_s

##
## methods to get lwc (for now, cloud only, i.e. up to 50um diameter). I'm
## also not considering ventilation corrections under the [as of now untested]
## assumption that they won't be significant for cloud size droplets)
## 
def get_lwc(cpd_dict, cutoff_bins):

    lwc_var_keys = ['lwc_5um_to_50um_diam']
    if not cutoff_bins:
        lwc_var_keys.append('lwc_sub_5um_diam')

    lwc = np.zeros(np.shape(cpd_dict['data']['time']))
    for lwc_var_key in lwc_var_keys:
        lwc += cpd_dict['data'][lwc_var_key]

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
