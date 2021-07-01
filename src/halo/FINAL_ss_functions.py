"""
Various routines and subroutines for calculating predicted 
superaturation from HALO campaign data
"""
import copy
import numpy as np
import re

from halo import CAS_bins, CDP_bins, CIP_bins, CAS_lo_bin, \
            CAS_mid_bin, CAS_up_bin, CIP_lo_bin, CIP_up_bin

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
LSR_INT = 0.04164662760767679
LSR_SLOPE = 0.8253679031561234

##
## methods to get ss_qss 
##
def get_ss_qss_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    full_spectrum_bin_radii = get_full_spectrum_bin_radii(CAS_bins, \
                                                    CIP_bins, 'log')

    meanr = get_meanr_vs_t(adlr_dict, full_spectrum_dict, \
            cutoff_bins, incl_rain, incl_vent, full_spectrum_bin_radii)
    nconc = get_nconc_vs_t(adlr_dict, full_spectrum_dict, \
            cutoff_bins, incl_rain, full_spectrum_bin_radii)

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

    return ss_qss

def get_ss_pred_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    ss_qss = get_ss_qss_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                    cutoff_bins, full_ss, incl_rain, incl_vent)
    ss_pred = LSR_INT + LSR_SLOPE*ss_qss

    return ss_pred

def get_ss_qss_components(adlr_dict, full_spectrum_dict, change_cas_corr, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    full_spectrum_bin_radii = get_full_spectrum_bin_radii(CAS_bins, \
                                                    CIP_bins, 'log')

    meanr = get_meanr_vs_t(adlr_dict, full_spectrum_dict, \
            cutoff_bins, incl_rain, incl_vent, full_spectrum_bin_radii)
    nconc = get_nconc_vs_t(adlr_dict, full_spectrum_dict, \
            cutoff_bins, incl_rain, full_spectrum_bin_radii)

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
    
    return A, B, meanr, nconc

##
## methods to combine CAS and CIP drop spectra into one dictionary 
##
def get_full_spectrum_dict(cas_dict, cip_dict, change_cas_corr):

    full_spectrum_dict = {'data': {}, 'units': {}}
    full_spectrum_bin_ind = 1

    for i in range(CAS_lo_bin, CAS_mid_bin):
        var_name = 'nconc_' + str(i)
        if change_cas_corr:
            var_name += '_corr'
        full_spectrum_dict['data']['nconc_' + str(full_spectrum_bin_ind)] = \
                                                    cas_dict['data'][var_name]
        full_spectrum_dict['units']['nconc_' + str(full_spectrum_bin_ind)] = 'm^-3'
        full_spectrum_bin_ind += 1

    overlap_cas_bins_nconc = np.zeros(np.shape(cas_dict['data']['time']))
    for i in range(CAS_mid_bin, CAS_up_bin):
        var_name = 'nconc_' + str(i)
        if change_cas_corr:
            var_name += '_corr'
        overlap_cas_bins_nconc += cas_dict['data'][var_name]

    dlogDp_CIP1 = np.log10(CIP_bins['upper'][0]/CIP_bins['lower'][0])
    dlogDp_CAS1216 = np.log10(CIP_bins['upper'][11]/CIP_bins['lower'][7])
    full_spectrum_dict['data']['nconc_' + str(full_spectrum_bin_ind)] = \
                        overlap_cas_bins_nconc*dlogDp_CIP1/dlogDp_CAS1216
    full_spectrum_dict['units']['nconc_' + str(full_spectrum_bin_ind)] = 'm^-3'
    full_spectrum_bin_ind += 1

    #exclude first CIP bin
    for i in range(2, CIP_up_bin+1):
        var_name = 'nconc_' + str(i)
        full_spectrum_dict['data']['nconc_' + str(full_spectrum_bin_ind)] = \
                                                    cip_dict['data'][var_name]
        full_spectrum_dict['units']['nconc_' + str(full_spectrum_bin_ind)] = 'm^-3'
        full_spectrum_bin_ind += 1

    return full_spectrum_dict

##
## methods to get meanr and nconc with rain (and optionally ventilation 
## corrections) for cas and cdp (cip included for both for higher radii)
##
def get_meanr_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, \
                incl_rain, incl_vent, full_spectrum_bin_radii):

    nconc = get_nconc_vs_t(adlr_dict, full_spectrum_dict, \
            cutoff_bins, incl_rain, full_spectrum_bin_radii)

    meanr_sum = np.zeros(np.shape(adlr_dict['data']['time']))

    for var_name in full_spectrum_dict['data'].keys():
        meanr_sum += get_meanr_contribution_from_spectrum_var(var_name, \
                            adlr_dict, full_spectrum_dict, cutoff_bins, \
                            incl_rain, incl_vent, full_spectrum_bin_radii)
    #print('innerm', np.nanmean(meanr_sum))
    
    return meanr_sum/nconc

def get_nconc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, \
                            incl_rain, full_spectrum_bin_radii):

    nconc_sum = np.zeros(np.shape(adlr_dict['data']['time']))

    for var_name in full_spectrum_dict['data'].keys():
        nconc_sum += get_nconc_contribution_from_spectrum_var(var_name, \
                            adlr_dict, full_spectrum_dict, cutoff_bins, \
                            incl_rain, full_spectrum_bin_radii)
    #print('innern', np.nanmean(nconc_sum))

    return nconc_sum

##
## methods to get meanr and nconc contributions from cas and cip dsd
## data. note: meanr contribution is weighted by nconc for that bin
##
def get_meanr_contribution_from_spectrum_var(var_name, adlr_dict, \
                        full_spectrum_dict, cutoff_bins, incl_rain, \
                        incl_vent, full_spectrum_bin_radii):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except IndexError: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 1
    bin_radius = full_spectrum_bin_radii[bin_ind]

    pres = adlr_dict['data']['pres']
    temp = adlr_dict['data']['temp']

    rho_air = pres/(R_a*temp)
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air**2./(eta**4.*g*rho_w) #pr&kl p 418

    u_term = get_u_term(bin_radius, eta, N_Be_div_r3, N_Bo_div_r2, \
                                            N_P, pres, rho_air, temp)
    N_Re = 2*rho_air*bin_radius*u_term/eta

    f = get_ventilation_coefficient(N_Re, incl_vent)

    nconc_contribution_from_var = get_nconc_contribution_from_spectrum_var( \
                                var_name, adlr_dict, full_spectrum_dict, \
                                cutoff_bins, incl_rain, full_spectrum_bin_radii)

    mean_r_contribution_from_var = nconc_contribution_from_var*bin_radius*f

    return mean_r_contribution_from_var

def get_nconc_contribution_from_spectrum_var(var_name, adlr_dict, \
                                    full_spectrum_dict, cutoff_bins, \
                                    incl_rain, full_spectrum_bin_radii):

    zero_arr = np.zeros(np.shape(adlr_dict['data']['time']))

    try:
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
    except IndexError: #there's no integer in var_name (not a bin variable)
        return zero_arr

    bin_ind = nconc_ind - 1
    bin_radius = full_spectrum_bin_radii[bin_ind]

    if cutoff_bins and bin_radius < 1.5e-6:
        return zero_arr
    elif not incl_rain and bin_radius > 50.e-6:
        return zero_arr
    else:
        return full_spectrum_dict['data']['nconc_' + str(nconc_ind)]

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
## method to get saturation vapor pressure
##
def get_sat_vap_pres(T):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return e_s

##
## methods to get lwc 
## 
def get_lwc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, rmax):

    full_spectrum_bin_radii = get_full_spectrum_bin_radii(CAS_bins, \
                                                    CIP_bins, 'log')

    lwc = np.zeros(np.shape(adlr_dict['data']['time']))
    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    for i, bin_radius in enumerate(full_spectrum_bin_radii):
        bin_ind = i + 1
        var_name = 'nconc_' + str(bin_ind)
        if cutoff_bins:
            if bin_radius < 1.5e-6:
                continue
        if bin_radius < rmax:
            lwc += 4./3.*np.pi*rho_w/rho_air*\
                full_spectrum_dict['data'][var_name]*bin_radius**3.

    return lwc

##
## methods to get the central radius values for each bin 
##
def get_full_spectrum_bin_radii(CAS_bins, CIP_bins, bin_scheme):

    if bin_scheme == 'lin':
        CAS_radii = (CAS_bins['upper'] + CAS_bins['lower'])/4.
        CIP_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.
    elif bin_scheme == 'log':
        CAS_radii = np.sqrt(CAS_bins['upper'] * CAS_bins['lower'])/2.
        CIP_radii = np.sqrt(CIP_bins['upper'] * CIP_bins['lower'])/2.
    else:
        print('incorrect bin scaling argument')
        return

    return splice_radii_arrays(CAS_radii, CIP_radii)

def splice_radii_arrays(CAS_radii, CIP_radii):

    full_spectrum_bin_radii = []

    for i in range(CAS_lo_bin, CAS_mid_bin):
        bin_ind = i - 5
        full_spectrum_bin_radii.append(CAS_radii[bin_ind])
    for i in range(CIP_lo_bin, CIP_up_bin+1):
        bin_ind = i - 1 
        full_spectrum_bin_radii.append(CIP_radii[bin_ind])

    return np.array(full_spectrum_bin_radii)

def get_full_spectrum_dlogDp(CAS_bins, CIP_bins):

    full_spectrum_dlogDp = []

    for i in range(CAS_lo_bin, CAS_mid_bin):
        bin_ind = i - 5
        full_spectrum_dlogDp.append(np.log10(\
            CAS_bins['upper'][bin_ind]/CAS_bins['lower'][bin_ind]))
    for i in range(CIP_lo_bin, CIP_up_bin+1):
        bin_ind = i - 1 
        full_spectrum_dlogDp.append(np.log10\
            (CIP_bins['upper'][bin_ind]/CIP_bins['lower'][bin_ind]))
    
    return np.array(full_spectrum_dlogDp)

def get_ss_coeff(pres, temp):

    rho_air = pres/(R_a*temp) 
    e_s = get_sat_vap_pres(temp)
    F_d = rho_w*R_v*temp/(D*e_s) 
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
    A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp*(F_d + F_k)
    B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_air*temp**2.)) 

    ss_coeff = A/(4*np.pi*B)*100. #as a percentage

    return ss_coeff

def get_ss_coeff_partial_dervs(pres, temp):

    min_pres = np.nanmin(pres)
    max_pres = np.nanmax(pres)
    if max_pres == min_pres:
        max_pres = 1.01*max_pres
    min_temp = np.nanmin(temp)
    max_temp = np.nanmax(temp)
    if max_temp == min_temp:
        max_temp = 1.01*max_temp

    pres_domain = np.linspace(min_pres, max_pres, 1000)
    pres_domain = np.append(pres_domain, \
                    max_pres + pres_domain[1] - pres_domain[0])
    temp_domain = np.linspace(min_temp, max_temp, 1000)
    temp_domain = np.append(temp_domain, \
                    max_temp + temp_domain[1] - temp_domain[0])

    P, T = np.meshgrid(pres_domain, temp_domain)

    rho_air = P/(R_a*T) 
    e_s = get_sat_vap_pres(T)
    F_d = rho_w*R_v*T/(D*e_s) 
    F_k = (L_v/(R_v*T) - 1)*L_v*rho_w/(K*T)
    A = g*(L_v*R_a/(C_ap*R_v)*1/T - 1)*1./R_a*1./T*(F_d + F_k)
    B = rho_w*(R_v*T/e_s + L_v**2./(R_v*C_ap*rho_air*T**2.)) 

    ss_coeff = A/(4*np.pi*B)*100. #as a percentage

    partial_ss_coeff = ss_coeff[:, 1:] - ss_coeff[:, :-1]
    partial_P = P[:, 1:] - P[:, :-1]
    dss_coeff_dP = partial_ss_coeff/partial_P

    partial_ss_coeff = ss_coeff[1:, :] - ss_coeff[:-1, :]
    partial_T = T[1:, :] - T[:-1, :]
    dss_coeff_dT = partial_ss_coeff/partial_T

    dss_coeff_dpres = np.zeros(np.shape(pres))
    dss_coeff_dtemp = np.zeros(np.shape(temp))

    for i, pres_val in enumerate(pres):
        temp_val = temp[i]
        pres_ind = get_grid_ind(pres_domain[:-1], pres_val)
        temp_ind = get_grid_ind(temp_domain[:-1], temp_val)
        dss_coeff_dpres[i] = dss_coeff_dP[pres_ind, temp_ind]
        dss_coeff_dtemp[i] = dss_coeff_dT[pres_ind, temp_ind]

    return dss_coeff_dtemp, dss_coeff_dpres 

def get_grid_ind(X, x_val, lower_ind=0):
    
    nX = np.shape(X)[0]
    if nX == 1:
        return lower_ind 

    dx = X[1] - X[0]
    search_ind = int(np.floor(nX/2))
    search_val = X[search_ind]

    if abs(x_val - search_val) < dx:
        return lower_ind+search_ind
    elif x_val < search_val:
        return get_grid_ind(X[:search_ind], x_val, lower_ind)
    else:
        return get_grid_ind(X[search_ind:], x_val, lower_ind+search_ind)
