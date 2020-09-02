"""
various routines and subroutines for calculating quasi-steady-state
superaturation from HALO campaign data
"""
import numpy as np
import re

from revhalo import CAS_bins, CDP_bins, CIP_bins

def get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, incl_rain, incl_vent, full_ss):

    if incl_rain:
        meanr = get_meanr_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins, incl_rain, incl_vent):
        nconc = get_nconc_vs_t_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins, incl_rain, incl_vent):
    else:
        meanr = get_meanr_vs_t_from_cas(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins):
        nconc = get_nconc_vs_t_from_cas(adlr_dict, cas_dict, cip_dict, \
                        change_cas_corr, cutoff_bins):

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

def get_ss_vs_t_cdp(adlr_dict, cdp_dict, cip_dict, \
                cutoff_bins, incl_rain, incl_vent, full_ss):

    if incl_rain:
        meanr = get_meanr_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent):
        nconc = get_nconc_vs_t_from_cdp_and_cip(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins, incl_rain, incl_vent):
    else:
        meanr = get_meanr_vs_t_from_cdp(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins):
        nconc = get_nconc_vs_t_from_cdp(adlr_dict, cdp_dict, cip_dict, \
                        cutoff_bins):

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


