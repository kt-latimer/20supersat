import numpy as np

###
### physical constants
###
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_e = 6.3781e6 #radius of Earth (m)
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

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

    return input_vars['QNCLOUD'][...]

def get_nconc_rain(input_vars):

    return input_vars['QNRAIN'][...]

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
