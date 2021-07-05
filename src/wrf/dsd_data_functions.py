"""
various methods for calculating quantities from WRF DSDs
"""
import numpy as np

from phys_consts import *
from wrf import WRF_bin_radii

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
N_Re_regime2_coeffs = [-0.318657e1, 0.992696, -0.153193e-2, \
                        -0.987059e-3, -0.578878e-3, 0.855176e-4, \
                        -0.327815e-5] #417
N_Re_regime3_coeffs = [-0.500015e1, 0.523778e1, -0.204914e1, \
                        0.475294, -0.542819e-1, 0.238449e-2] #418

def get_bin_nconc(i, input_vars, rho_air_data):

    r_i = WRF_bin_radii[i-1]
    wrf_var_name = 'ff1i'+f'{i:02}'
    wrf_var_data = input_vars[wrf_var_name][...]
    nconc_data = wrf_var_data/(4./3.*np.pi*r_i**3.*rho_w/rho_air_data)

    return nconc_data

def get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                        pres_data, rho_air_data, temp_data):

    r_i = WRF_bin_radii[i-1]
    u_term_i = get_u_term(r_i, eta, N_Be_div_r3, N_Bo_div_r2, \
                            N_P, pres_data, rho_air_data, temp_data)
    N_Re_i = 2*rho_air_data*r_i*u_term_i/eta 
    f_data = get_vent_coeff(N_Re_i)

    return f_data

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

def get_vent_coeff(N_Re):
    """
    get ventilation coefficient (from pruppacher and klett p 541)
    """
    f = np.piecewise(N_Re, [N_Re < 2.46, N_Re >= 2.46], \
                    [lambda N_Re: 1. + 0.086*N_Re, \
                    lambda N_Re: 0.78 + 0.27*N_Re**0.5])
    return f
