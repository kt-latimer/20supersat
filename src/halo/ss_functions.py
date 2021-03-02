"""
Various routines and subroutines for calculating predicted 
superaturation from HALO campaign data
"""
import numpy as np
import re

from halo import CAS_bins, CDP_bins, CIP_bins

##
## center radii of bins
##
CAS_bin_radii = (CAS_bins['upper'] + CAS_bins['lower'])/4.
CDP_bin_radii = (CDP_bins['upper'] + CDP_bins['lower'])/4.
CIP_bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

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
## least squares regression from wrf data 
## 
LSR_INT = 0.011320347049628032
LSR_SLOPE = 0.8498781398164678

##
## methods to get ss_qss for cas and cdp
##
def get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent):

    if incl_rain:
        meanr = get_meanr_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins, incl_rain, incl_vent)
        nconc = get_nconc_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins, incl_rain, incl_vent)
    else:
        meanr = get_meanr_vs_t_from_cas(adlr_dict, cas_dict, \
                                    change_cas_corr, cutoff_bins, \
                                    incl_rain, incl_vent)
        nconc = get_nconc_vs_t_from_cas(adlr_dict, cas_dict, \
                                    change_cas_corr, cutoff_bins, \
                                    incl_rain, incl_vent)

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    if full_ss:
        pres = adlr_dict['data']['pres']
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
    ss_pred = LSR_INT + LSR_SLOPE*ss_qss

    return ss_pred

def get_ss_vs_t_cdp(adlr_dict, cdp_dict, cip_dict, \
                cutoff_bins, full_ss, incl_rain, incl_vent):

    if incl_rain:
        meanr = get_meanr_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent)
        nconc = get_nconc_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent)
    else:
        meanr = get_meanr_vs_t_from_cdp(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent)
        nconc = get_nconc_vs_t_from_cdp(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent)

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    if full_ss:
        pres = adlr_dict['data']['pres']
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
    ss_pred = LSR_INT + LSR_SLOPE*ss_qss

    return ss_pred

##
## methods to get meanr and nconc with rain (and optionally ventilation 
## corrections) for cas and cdp (cip included for both for higher radii)
##
def get_meanr_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, incl_rain, incl_vent):

    nconc = get_nconc_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                    change_cas_corr, cutoff_bins, incl_rain, incl_vent)

    meanr_sum = np.zeros(np.shape(cas_dict['data']['time']))
    print(np.shape(meanr_sum))

    for var_name in cas_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cas_var(var_name, \
                            adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent)
    
    for var_name in cip_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                            cip_dict, incl_rain, incl_vent)

    return meanr_sum/nconc

def get_nconc_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, incl_rain, incl_vent):

    nconc_sum = np.zeros(np.shape(cas_dict['data']['time']))

    for var_name in cas_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cas_var(var_name, \
                            adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent)

    for var_name in cip_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
                            cip_dict)

    return nconc_sum

def get_meanr_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                cutoff_bins, incl_rain, incl_vent):

    nconc = get_nconc_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                    cutoff_bins, incl_rain, incl_vent)

    meanr_sum = np.zeros(np.shape(cdp_dict['data']['time']))

    for var_name in cas_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cdp_var(var_name, adlr_dict, \
                            cdp_dict, cutoff_bins, incl_rain, incl_vent)
    
    for var_name in cip_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                            cip_dict, incl_rain, incl_vent)

    return meanr_sum/nconc

def get_nconc_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                cutoff_bins, incl_rain, incl_vent):

    nconc_sum = np.zeros(np.shape(cdp_dict['data']['time']))

    for var_name in cdp_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cdp_var(var_name, adlr_dict, \
                    cdp_dict, cutoff_bins, incl_rain, incl_vent)

    for var_name in cip_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
                    cip_dict)

    return nconc_sum

##
## methods to get meanr and nconc without rain for cas and cdp
##
def get_meanr_vs_t_from_cas(adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent):

    nconc = get_nconc_vs_t_from_cas(adlr_dict, cas_dict, \
                    change_cas_corr, cutoff_bins, incl_rain, incl_vent)

    meanr_sum = np.zeros(np.shape(cas_dict['data']['time']))

    for var_name in cas_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cas_var(var_name, \
                            adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent)

    return meanr_sum/nconc

def get_nconc_vs_t_from_cas(adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent):

    nconc_sum = np.zeros(np.shape(cas_dict['data']['time']))

    for var_name in cas_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cas_var(var_name, \
                            adlr_dict, cas_dict, change_cas_corr, \
                            cutoff_bins, incl_rain, incl_vent)

    return nconc_sum

def get_meanr_vs_t_from_cdp(adlr_dict, cdp_dict, cutoff_bins, \
                            incl_rain, incl_vent):

    nconc = get_nconc_vs_t_from_cdp(adlr_dict, cdp_dict, cutoff_bins, \
                                    incl_rain, incl_vent)

    meanr_sum = np.zeros(np.shape(cdp_dict['data']['time']))

    for var_name in cas_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_cdp_var(var_name, adlr_dict, \
                            cdp_dict, cutoff_bins, incl_rain, incl_vent)

    return meanr_sum/nconc

def get_nconc_vs_t_from_cdp(adlr_dict, cdp_dict, cutoff_bins):

    nconc_sum = np.zeros(np.shape(cdp_dict['data']['time']))

    for var_name in cdp_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_cdp_var(var_name, adlr_dict, \
                    cdp_dict, cutoff_bins, incl_rain, incl_vent)

    return nconc_sum

##
## methods to get meanr and nconc contributions from cas, cdp, and cip
## dsd data. note: meanr contribution is weighted by nconc for that bin
##
def get_meanr_contribution_from_cas_var(var_name, adlr_dict, cas_dict, \
                                        change_cas_corr, cutoff_bins, \
                                        incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    if 'diam' in var_name:
        return zero_arr
    
    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except IndexError: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 5
    r = CAS_bin_radii[bin_ind]

    pres = adlr_dict['data']['pres']
    temp = adlr_dict['data']['temp']

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

    nconc_contribution_from_var = get_nconc_contribution_from_cas_var( \
        var_name, adlr_dict, cas_dict, change_cas_corr, cutoff_bins, \
        incl_rain, incl_vent)

    mean_r_contribution_from_var = nconc_contribution_from_var*r*f

    return mean_r_contribution_from_var

def get_meanr_contribution_from_cdp_var(var_name, adlr_dict, \
                     cdp_dict, cutoff_bins, incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    if 'diam' in var_name:
        return zero_arr
    
    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except indexerror: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 1
    r = CDP_bin_radii[bin_ind]

    pres = adlr_dict['data']['pres']
    temp = adlr_dict['data']['temp']

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

    nconc_contribution_from_var = get_nconc_contribution_from_cdp_var( \
        var_name, adlr_dict, cdp_dict, cutoff_bins, incl_rain, incl_vent)

    mean_r_contribution_from_var = nconc_contribution_from_var*r*f

    return mean_r_contribution_from_var

def get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                                        cip_dict, incl_rain, \
                                        incl_vent):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))
    
    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except IndexError: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 1
    r = CIP_bin_radii[bin_ind]

    pres = adlr_dict['data']['pres']
    temp = adlr_dict['data']['temp']

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

    nconc_contribution_from_var = get_nconc_contribution_from_cip_var( \
        var_name, adlr_dict, cip_dict)

    mean_r_contribution_from_var = nconc_contribution_from_var*r*f

    return mean_r_contribution_from_var

def get_nconc_contribution_from_cas_var(var_name, adlr_dict, cas_dict, \
                        change_cas_corr, cutoff_bins, incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))
    
    is_bin_var = check_if_bin_var(var_name)
    if not is_bin_var:
        return zero_arr 

    has_correct_correction_factor = \
        check_if_correct_correction_factor(var_name, change_cas_corr)
    if not has_correct_correction_factor:
        return zero_arr 

    has_correct_lower_bin_cutoff = \
        has_correct_lower_bin_cutoff_cas(var_name, cutoff_bins)
    if not has_correct_lower_bin_cutoff:
        return zero_arr 

    has_correct_upper_bin_cutoff = \
        has_correct_upper_bin_cutoff_cas(var_name, incl_rain)
    if not has_correct_upper_bin_cutoff:
        return zero_arr 

    nconc_contribution_from_var = cas_dict['data'][var_name]

    return nconc_contribution_from_var
    
def get_nconc_contribution_from_cdp_var(var_name, adlr_dict, \
                                        cdp_dict, cutoff_bins, \
                                        incl_rain, incl_vent):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    is_bin_var = check_if_bin_var(var_name)
    if not is_bin_var:
        return zero_arr 

    has_correct_lower_bin_cutoff = \
        has_correct_lower_bin_cutoff_cdp(var_name, cutoff_bins)
    if not has_correct_lower_bin_cutoff:
        return zero_arr 

    has_correct_upper_bin_cutoff = \
        has_correct_upper_bin_cutoff_cdp(var_name, incl_rain)
    if not has_correct_upper_bin_cutoff:
        return zero_arr 

    nconc_contribution_from_var = cdp_dict['data'][var_name]

    return nconc_contribution_from_var

def get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
                                        cip_dict):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    is_bin_var = check_if_bin_var(var_name)
    if not is_bin_var:
        return zero_arr 

    nconc_contribution_from_var = cip_dict['data'][var_name]

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
        u_term = (1 + 1.26*lam/r)*(2*r**2.*g*rho_w/(9*eta))
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

    if 'nconc' in var_name and 'tot' not in var_name:
        return True
    else:
        return False

def check_if_correct_correction_factor(var_name, change_cas_corr):
    
    if change_cas_corr and 'corr' in var_name:
        return True
    elif not change_cas_corr and 'corr' not in var_name:
        return True
    else:
        return False

def has_correct_lower_bin_cutoff_cas(var_name, cutoff_bins):

    if not cutoff_bins:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 5
    lower_bin_radius = CAS_bins['lower'][bin_ind]
    
    if lower_bin_radius >= 5.e-6:
        return True
    else:
        return False

def has_correct_upper_bin_cutoff_cas(var_name, incl_rain):

    if not incl_rain:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 5
    upper_bin_radius = CAS_bins['upper'][bin_ind]
    
    if upper_bin_radius <= 25.e-6: 
        return True
    else:
        return False

def has_correct_lower_bin_cutoff_cdp(var_name, cutoff_bins):

    if not incl_rain:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 1
    lower_bin_radius = CDP_bins['lower'][bin_ind]
    
    if lower_bin_radius >= 5.e-6:
        return True
    else:
        return False

def has_correct_upper_bin_cutoff_cdp(var_name, incl_rain):

    if not cutoff_bins:
        return True
    
    nconc_ind = int(re.findall(r'\d+', var_name)[0])
    bin_ind = nconc_ind - 1
    upper_bin_radius = CDP_bins['upper'][bin_ind]
    
    if upper_bin_radius <= 24.6e-6: #not perfect but just goin with it... 
        return True
    else:
        return False

##
## method to get saturation vapor pressure
##
def get_sat_vap_pres(T):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return e_s

##
## methods to get lwc (for now, cloud only, i.e. up to either top bin in 
## cas / cdp. this is imperfect in the case where incl_rain is True but
## there doesn't seem to be a clear best choice in this case. I'm also
## not considering ventilation corrections under the [as of now untested]
## assumption that they won't be significant for cloud size droplets)
## 
def get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins):

    lwc_var_keys = ['lwc_5um_to_25um_diam', 'lwc_above_25um_diam']
    if not cutoff_bins:
        lwc_var_keys.append('lwc_sub_5um_diam')
    if change_cas_corr:
        lwc_var_keys = [lwc_var_key+'_corr' for lwc_var_key in lwc_var_keys]

    lwc = np.zeros(np.shape(cas_dict['data']['time']))
    for lwc_var_key in lwc_var_keys:
        lwc += cas_dict['data'][lwc_var_key]

    return lwc

def get_lwc_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, change_cas_corr, cutoff_bins):

    lwc_cas = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
    
    #the rest is a bit brute force / ad hoc because there's only two \
    #bins to use from the cip data and both need to get truncated.
    cip_nconc_1 = get_cip_nconc_1(cas_dict, cip_dict, change_cas_corr)
    cip_radius_1 = (CAS_bin_radii[-1] + CIP_bin_radii[1])/2.
    print(cip_radius_1)
    cip_nconc_2 = get_cip_nconc_2(cip_dict)
    cip_radius_2 = (CIP_bin_radii[1] + 51e-6)/2.
    print(cip_radius_2)

    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    lwc_cip = 4./3.*np.pi*rho_w/rho_air*(cip_nconc_1*cip_radius_1**3. + \
                                            cip_nconc_2*cip_radius_2**3.)
    return lwc_cas + lwc_cip

def get_cip_nconc_1(cas_dict, cip_dict, change_cas_corr):

    cas_nconc_above_25um_diam = np.zeros( \
        np.shape(cas_dict['data']['time']))

    for i in range(12, 17):
        bin_ind = i - 5
        var_key = 'nconc_' + str(i)
        if change_cas_corr:
            var_key += '_corr'
        cas_nconc_above_25um_diam += \
                cas_dict['data'][var_key]*CAS_bin_radii[bin_ind]**3.

    cip_nconc_1 = cip_dict['data']['nconc_1'] - cas_nconc_above_25um_diam

    num_below_zero = np.sum(cip_nconc_1 < 0)
    if num_below_zero > 0:
        print(str(num_below_zero) + ' below zero')
        print(np.shape(cip_nconc_1))

    return cip_nconc_1 

def get_cip_nconc_2(cip_dict):

    new_bin_2_width = (51e-6 - CIP_bin_radii[1])
    old_bin_2_width = (CIP_bin_radii[2] - CIP_bin_radii[1])
    bin_2_width_fraction = new_bin_2_width/old_bin_2_width

    return bin_2_width_fraction*cip_dict['data']['nconc_2']

def get_lwc_from_cdp(cdp_dict, cutoff_bins):

    lwc_var_keys = ['lwc_5um_to_25um_diam', 'lwc_above_25um_diam']
    if not cutoff_bins:
        lwc_var_keys.append('lwc_sub_5um_diam')

    lwc = np.zeros(np.shape(cas_dict['data']['time']))
    for lwc_var_key in lwc_var_keys:
        lwc += cas_dict['data'][lwc_var_key]

    return lwc
