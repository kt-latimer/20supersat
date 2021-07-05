import numpy as np

from phys_consts import *

###
### wrf reference values
###
P_0 = 1.e5 #ref pressure (Pa)
T_0 = 300. #ref pot temp in (K)

def get_A(input_vars):

    temp = get_temp(input_vars)
    e_sat = get_e_sat(input_vars, temp=temp)
    
    F_d = rho_w*R_v*temp/(D*e_sat)
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)

    A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp*(F_d + F_k)
    
    return A

def get_B(input_vars):

    temp = get_temp(input_vars)
    e_sat = get_e_sat(input_vars, temp=temp)
    rho_air = get_rho_air(input_vars, temp=temp)
    
    B = rho_w*(R_v*temp/e_sat + L_v**2./(R_v*C_ap*rho_air*temp**2.))

    return B

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

def get_e_sat(input_vars, temp=None):

    if temp is None:
        temp = get_temp(input_vars)

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

def get_LH(input_vars):

    return input_vars['TEMPDIFFL'][...]

def get_LWC_cloud(input_vars):

    return input_vars['QCLOUD'][...]

def get_LWC_rain(input_vars):

    return input_vars['QRAIN'][...]

def get_nconc_cloud(input_vars):

    rho_air = get_rho_air(input_vars)
    return input_vars['QNCLOUD'][...]*rho_air

def get_nconc_rain(input_vars):

    rho_air = get_rho_air(input_vars)
    return input_vars['QNRAIN'][...]*rho_air

def get_pres(input_vars):

    PB = input_vars['PB'][...]
    P = input_vars['P'][...]
    pres = PB + P

    return pres

def get_rho_air(input_vars, pres=None, temp=None):

    if pres is None:
        pres = get_pres(input_vars)

    if temp is None:
        temp = get_temp(input_vars, pres=pres)

    rho_air = pres/(R_a*temp)

    return rho_air

def get_rain_rate(input_vars):

    return input_vars['PRECR3D'][...]

def get_ss_wrf(input_vars):

    return input_vars['SSW'][...]

def get_temp(input_vars, pres=None):

    if pres is None:
        pres = get_pres(input_vars)

    T = input_vars['T'][...]

    theta = T_0 + T 
    temp = theta*np.power((pres/P_0), R_a/C_ap)

    return temp

def get_w(input_vars):

    w_staggered = input_vars['W'][...]
    #vertical wind velocity is on a staggered grid; take NN average to
    #reshape to mass grid
    w = (w_staggered[:,0:-1,:,:] + w_staggered[:,1:,:,:])/2.

    return w

def get_x(input_vars):

    XLONG = input_vars['XLONG'][...] #longitude
    x = XLONG*np.pi/180.*R_e

    return x

def get_y(input_vars):

    XLAT = input_vars['XLAT'][...] #latitude
    y = XLAT*np.pi/180.*R_e

    return y 

def get_z(input_vars):

    PH = input_vars['PH'][...]
    PHB = input_vars['PHB'][...]
    z = (PH + PHB)/g #altitude rel to sea level
    #reshape to mass grid
    z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2.

    return z
