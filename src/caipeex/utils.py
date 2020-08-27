"""
various utility methods for halo package.
"""
from itertools import product
import numpy as np

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR

dsdbinfile = DATA_DIR + 'DSD_bins.npy'
DSD_bins = np.load(dsdbinfile, allow_pickle=True).item()
centr_dsd = (DSD_bins['upper'] + DSD_bins['lower'])/4. #diam to radius
dr_dsd = DSD_bins['upper'] - DSD_bins['lower']
nbins_dsd = len(centr_dsd)

#physical constants
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

#various series expansion coeffs - comment = page in pruppacher and klett
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130
N_Re_regime2_coeffs = [-0.318657e1, 0.992696, -0.153193e-2, \
                        -0.987059e-3, -0.578878e-3, 0.855176e-4, \
                        -0.327815e-5] #417
N_Re_regime3_coeffs = [-0.500015e1, 0.523778e1, -0.204914e1, \
                        0.475294, -0.542819e-1, 0.238449e-2] #418

def get_meanr(dataset, cutoff_bins):
    nconc = get_nconc(dataset, cutoff_bins) 
    radsum = np.zeros(dataset['data']['time'].shape)
    if cutoff_bins:
        low_bin_dsd = 4
    else:
        low_bin_dsd = 1

    for i in range(low_bin_dsd, 31):
        var_key = 'nconc_' + str(i)
        radsum += centr_dsd[i-1]*dataset['data'][var_key]
    return radsum/nconc

def get_nconc(dataset, cutoff_bins):
    nconc = np.zeros(dataset['data']['time'].shape)
    if cutoff_bins:
        low_bin_dsd = 4
    else:
        low_bin_dsd = 1

    for i in range(low_bin_dsd, 31):
        var_key = 'nconc_' + str(i)
        nconc += dataset['data'][var_key]
    return nconc

def get_meanfr_inclrain(dataset, metdata, cutoff_bins):
    pres = metdata['data']['pres']
    temp = metdata['data']['temp']
    w = metdata['data']['vert_wind_vel']

    rho_a = pres/(R_a*temp)
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_a*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_a**2./(eta**4.*g*rho_w) #pr&kl p 418

    u_term = np.array([get_u_term(r, eta, N_Be_div_r3, N_Bo_div_r2, \
                            N_P, pres, rho_a, temp) for r in centr_dsd])
    N_Re_vals = np.array([2*rho_a*r*u_term[j]/eta for j, r in enumerate(centr_dsd)])
    f_vals = np.array([get_vent_coeff(N_Re) for N_Re in N_Re_vals])

    nconc = get_nconc_inclrain(dataset, cutoff_bins) 
    ventradsum = np.zeros(dataset['data']['time'].shape)

    if cutoff_bins:
        low_bin_dsd = 4
    else:
        low_bin_dsd = 1

    for i in range(low_bin_dsd, 92):
        var_key = 'nconc_' + str(i)
        ventradsum += f_vals[i-1]*centr_dsd[i-1]*dataset['data'][var_key]
    return ventradsum/nconc

def get_meanr_inclrain(dataset, cutoff_bins):
    nconc = get_nconc_inclrain(dataset, cutoff_bins) 
    radsum = np.zeros(dataset['data']['time'].shape)

    if cutoff_bins:
        low_bin_dsd = 4
    else:
        low_bin_dsd = 1

    for i in range(low_bin_dsd, 92):
        var_key = 'nconc_' + str(i)
        radsum += centr_dsd[i-1]*dataset['data'][var_key]
    return radsum/nconc

def get_nconc_inclrain(dataset, cutoff_bins):
    nconc = np.zeros(dataset['data']['time'].shape)

    if cutoff_bins:
        low_bin_dsd = 4
    else:
        low_bin_dsd = 1

    for i in range(low_bin_dsd, 92):
        var_key = 'nconc_' + str(i)
        nconc += dataset['data'][var_key]
    return nconc

def get_ss_full(dataset, metdata, cutoff_bins):

    meanr = get_meanr(dataset, cutoff_bins)
    nconc = get_nconc(dataset, cutoff_bins)
    pres = metdata['data']['pres']
    temp = metdata['data']['temp']
    w = metdata['data']['vert_wind_vel']

    rho_a = pres/(R_a*temp)
    A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
    e_s = get_sat_vap_pres(temp)
    B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_a*temp**2.))
    F_d = rho_w*R_v*temp/(D*e_s) 
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
    ss = A*w*(F_d + F_k)/(4*np.pi*nconc*meanr*B)*100

    return (np.array(ss))

def get_ss_full_inclrain(dataset, metdata, cutoff_bins):

    meanr = get_meanr_inclrain(dataset, cutoff_bins)
    nconc = get_nconc_inclrain(dataset, cutoff_bins)
    pres = metdata['data']['pres']
    temp = metdata['data']['temp']
    w = metdata['data']['vert_wind_vel']

    rho_a = pres/(R_a*temp)
    A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
    e_s = get_sat_vap_pres(temp)
    B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_a*temp**2.))
    F_d = rho_w*R_v*temp/(D*e_s) 
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
    ss = A*w*(F_d + F_k)/(4*np.pi*nconc*meanr*B)*100

    return (np.array(ss))

def get_ss_full_inclrain_and_vent(dataset, metdata, cutoff_bins):

    meanfr = get_meanfr_inclrain(dataset, metdata, cutoff_bins)
    nconc = get_nconc_inclrain(dataset, cutoff_bins)
    pres = metdata['data']['pres']
    temp = metdata['data']['temp']
    w = metdata['data']['vert_wind_vel']

    rho_a = pres/(R_a*temp)
    A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
    e_s = get_sat_vap_pres(temp)
    B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_a*temp**2.))
    F_d = rho_w*R_v*temp/(D*e_s) 
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
    ss = A*w*(F_d + F_k)/(4*np.pi*nconc*meanfr*B)*100
    return ss

def get_sat_vap_pres(T):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return e_s

def get_u_term(r, eta, N_Be_div_r3, N_Bo_div_r2, N_P, pres, rho_a, temp):
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
        u_term = eta*N_Re/(2*rho_a*r)
    else:
        N_Bo = N_Bo_div_r2*r**2.
        X = np.log(16./3.*N_Bo*N_P**(1./6.))
        N_Re = N_P**(1./6.)*np.exp(sum([N_Re_regime3_coeffs[i]*X**i for i in \
                                    range(len(N_Re_regime3_coeffs))]))
        u_term = eta*N_Re/(2*rho_a*r)
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

def get_vent_coeff(N_Re):
    """
    get ventilation coefficient (from pruppacher and klett p 541)
    """
    f = np.piecewise(N_Re, [N_Re < 2.46, N_Re >= 2.46], \
                    [lambda N_Re: 1. + 0.086*N_Re, \
                    lambda N_Re: 0.78 + 0.27*N_Re**0.5])
    return f

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

