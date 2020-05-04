"""
Create and save file in DATA_DIR with secondary variables derived from raw WRF
output. Two files are created; one for BG, one for PI. Each file contains the
variables 'lh_J_m3_s' (latent heating by water vapor diffusion; J/(m^3 s)),
'lh_K_s' (latent heating by water vapor diffusion; K/s), 'lwc_cloud' (liquid 
water content in cloud droplets; g/g), 'lwc_rain' (lwc in rain drops; g/g), 
'lwc_tot' (lwc in cloud and rain; g/g), 'meanfr' (mean radius-weighted
ventilation factor; m), 'meanr' (mean cloud drop radius; m), 'meanucloud'
(number-weighted fall speed of cloud droplets; m/s), 'meanurain'
(number-weighted fall speed of rain drops; m/s), 'meanur3cloud' (mass-weighted
fall speed of cloud droplets; m/s), 'meanur3rain' (mass-weighted fall speed of
rain droplets; m/s), 'nconccloud' (cloud droplet number concentration; m^-3),
'nconcrain' (rain drop number concentration; m^-3), 'nconc' (cloud and rain 
drop number concentration; m^-3), 'pres' (total pressure; Pa), 'rho_air' (dry 
air density from ideal gas law; kg/m^3), 'ss_qss' (quasi-steady-state 
supersaturation, no ventilation), 'ss_ry' (supersaturation from condensation
rate as given in Rogers and Yau, no ventilation), 'ss_wrf' (supersaturation
from WRF output), 'temp' (total temperature; K), 'theta' (total potential 
temperature; K), and 'w' (vertical wind velocity averaged to the centered 
(mass) grid; m/s). 
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

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

#wrf reference values
P_0 = 1.e5 #ref pressure (Pa)
T_0 = 300. #ref pot temp in (K)

#various series expansion coeffs - comment = page in pruppacher and klett
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130
N_Re_regime2_coeffs = [-0.318657e1, 0.992696, -0.153193e-2, \
                        -0.987059e-3, -0.578878e-3, 0.855176e-4, \
                        -0.327815e-5] #417
N_Re_regime3_coeffs = [-0.500015e1, 0.523778e1, -0.204914e1, \
                        0.475294, -0.542819e-1, 0.238449e-2] #418

def main():
    """
    For both polluted and unpolluted model runs, create and save netCDF files.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load primary datafiles
        ncinfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncinfile.variables
        
        #create secondary datafile
        ncoutfile = Dataset(DATA_DIR + model_dir \
                        + 'wrfout_d01_secondary_vars_with_rain_and_vent', 'w')

        #make file dimensions
        ncoutfile.createDimension('west_east', 450)
        ncoutfile.createDimension('south_north', 450)
        ncoutfile.createDimension('bottom_top', 66)
        ncoutfile.createDimension('Time', 84)
        dims = ('Time', 'bottom_top', 'south_north', 'west_east')

        #get primary variables from wrf output
        lh_K_s_data = ncprimvars['TEMPDIFFL'][...]
        lwc_cloud_data = ncprimvars['QCLOUD'][...]
        lwc_rain_data = ncprimvars['QRAIN'][...]
        PB = ncprimvars['PB'][...]
        P = ncprimvars['P'][...]
        ss_wrf_data = ncprimvars['SSW'][...]
        T = ncprimvars['T'][...]
        W = ncprimvars['W'][...]
        #vertical wind velocity is on a staggered grid; take NN average to
        #reshape to mass grid
        w_data = (W[:,0:-1,:,:] + W[:,1:,:,:])/2

        #calculate secondary variables
        theta_data = T_0 + T 
        pres_data = PB + P
        del PB, P, T, W #for memory
        temp_data = theta_data*np.power((pres_data/P_0), R_a/C_ap)
        rho_air_data = pres_data/(R_a*temp_data)
        
        eta = get_dyn_visc(temp_data)
        sigma = sum([sigma_coeffs[i]*(temp_data - 273)**i for i in \
                    range(len(sigma_coeffs))])*1.e-3
        N_Be_div_r3 = 32*rho_w*rho_air_data*g/(3*eta**2.) #pr&kl p 417
        N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
        N_P = sigma**3.*rho_air_data**2./(eta**4.*g*rho_w) #pr&kl p 418

        #compute mean radius of cloud droplets
        diams = [4*(2.**(i/3.))*10**(-6) for i in range(33)] #bin diams in m
        rads = [d/2. for d in diams]
        
        fr_wtsum = np.empty(np.shape(pres_data))
        r_wtsum = np.empty(np.shape(pres_data))
        u_cloud_wtsum = np.empty(np.shape(pres_data))
        u_rain_wtsum = np.empty(np.shape(pres_data))
        ur3_cloud_wtsum = np.empty(np.shape(pres_data))
        ur3_rain_wtsum = np.empty(np.shape(pres_data))
        N_sum = np.empty(np.shape(pres_data))
        Ncloud_sum = np.empty(np.shape(pres_data))
        Nrain_sum = np.empty(np.shape(pres_data))

        for i in range(33):
            r_i = rads[i]
            ff_i_wrf = ncprimvars['ff1i'+f'{i+1:02}'][...]
            N_i = ff_i_wrf/(4./3.*np.pi*r_i**3.*rho_w/rho_air_data)
            N_sum += N_i
            u_term_i = get_u_term(r_i, eta, N_Be_div_r3, N_Bo_div_r2, \
                                    N_P, pres_data, rho_air_data, temp_data)
            N_Re_i = 2*rho_air_data*r_i*u_term_i/eta 
            f_i = get_vent_coeff(N_Re_i)
            fr_wtsum += N_i*r_i*f_i
            r_wtsum += N_i*r_i
            if i < 15:
                u_cloud_wtsum += N_i*u_term_i
                ur3_cloud_wtsum += N_i*u_term_i*r_i**3.
                Ncloud_sum += N_i
            else:
                u_rain_wtsum += N_i*u_term_i
                ur3_rain_wtsum += N_i*u_term_i*r_i**3.
                Nrain_sum += N_i
            del ff_i_wrf, N_i #for memory

        meanfr_data = fr_wtsum/N_sum
        meanr_data = r_wtsum/N_sum 
        meanucloud_data = u_cloud_wtsum/Ncloud_sum
        meanurain_data = u_rain_wtsum/Nrain_sum
        meanur3cloud_data = ur3_cloud_wtsum/Ncloud_sum
        meanur3rain_data = ur3_rain_wtsum/Nrain_sum
        nconc_data = N_sum
        nconccloud_data = Ncloud_sum
        nconcrain_data = Nrain_sum

        #for memory
        del N_sum, Ncloud_sum, Nrain_sum, eta, fr_wtsum, r_wtsum, \
            u_cloud_wtsum, u_rain_wtsum, ur3_cloud_wtsum, ur3_rain_wtsum, \
            N_Be_div_r3, N_Bo_div_r2, N_P, sigma

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp_data - 1)*1./R_a*1./temp_data
        ss_qss_data = w_data*A/(4*np.pi*D*nconc_data*meanr_data)
        del A #for memory
        e_s = 611.2*np.exp(17.67*(temp_data - 273)/(temp_data - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        F_k = (L_v/(R_v*temp_data) - 1)*(L_v*rho_w/(K*temp_data))
        F_d = rho_w*R_v*temp_data/(D*e_s)

        #ss formula from RY p 102
        ss_ry_data = C_ap*rho_air_data \
            *(F_k + F_d)*lh_K_s_data/(4*np.pi*L_v*rho_w*meanr_data*nconc_data)
        del e_s, F_d, F_k #for memory

        #make variables for netCDF file
        lh_K_s= ncoutfile.createVariable('lh_K_s', np.dtype('float32'), dims)
        lh_J_m3_s = ncoutfile.createVariable('lh_J_m3_s', np.dtype('float32'), dims)
        lwc_cloud = ncoutfile.createVariable('lwc_cloud', np.dtype('float32'), dims)
        lwc_rain = ncoutfile.createVariable('lwc_rain', np.dtype('float32'), dims)
        lwc_tot = ncoutfile.createVariable('lwc_tot', np.dtype('float32'), dims)
        meanfr = ncoutfile.createVariable('meanfr', np.dtype('float32'), dims)
        meanr = ncoutfile.createVariable('meanr', np.dtype('float32'), dims)
        meanucloud = ncoutfile.createVariable('meanucloud', np.dtype('float32'), dims)
        meanurain = ncoutfile.createVariable('meanurain', np.dtype('float32'), dims)
        meanur3cloud = ncoutfile.createVariable('meanur3cloud', np.dtype('float32'), dims)
        meanur3rain = ncoutfile.createVariable('meanur3rain', np.dtype('float32'), dims)
        nconc = ncoutfile.createVariable('nconc', np.dtype('float32'), dims)
        nconccloud = ncoutfile.createVariable('nconccloud', np.dtype('float32'), dims)
        nconcrain = ncoutfile.createVariable('nconcrain', np.dtype('float32'), dims)
        pres = ncoutfile.createVariable('pres', np.dtype('float32'), dims)
        rho_air = ncoutfile.createVariable('rho_air', np.dtype('float32'), dims)
        ss_qss = ncoutfile.createVariable('ss_qss', np.dtype('float32'), dims)
        ss_ry = ncoutfile.createVariable('ss_ry', np.dtype('float32'), dims)
        ss_wrf = ncoutfile.createVariable('ss_wrf', np.dtype('float32'), dims)
        temp = ncoutfile.createVariable('temp', np.dtype('float32'), dims)
        theta = ncoutfile.createVariable('theta', np.dtype('float32'), dims)
        w = ncoutfile.createVariable('w', np.dtype('float32'), dims)
        
        #write data to variables
        lh_K_s[...] = lh_K_s_data
        lh_J_m3_s[...] = lh_K_s_data*C_ap*rho_air_data
        lwc_cloud[...] = lwc_cloud_data
        lwc_rain[...] = lwc_rain_data
        lwc_tot[...] = lwc_cloud_data + lwc_rain_data
        meanfr[...] = meanfr_data
        meanr[...] = meanr_data
        meanucloud[...] = meanucloud_data
        meanurain[...] = meanurain_data
        meanur3cloud[...] = meanur3cloud_data
        meanur3rain[...] = meanur3rain_data
        nconc[...] = nconc_data
        nconccloud[...] = nconccloud_data
        nconcrain[...] = nconcrain_data
        pres[...] = pres_data
        rho_air[...] = rho_air_data
        ss_qss[...] = ss_qss_data
        ss_ry[...] = ss_ry_data
        ss_wrf[...] = ss_wrf_data
        temp[...] = temp_data
        theta[...] = theta_data
        w[...] = w_data

        del lh_K_s_data, lwc_cloud_data, lwc_rain_data, meanr_data, \
            meanfr_data, meanucloud_data, meanur3cloud_data, meanurain_data, \
            meanur3rain_data, nconccloud_data, nconcrain_data, \
            nconc_data, pres_data, rho_air_data, ss_qss_data, ss_ry_data, \
            ss_wrf_data, temp_data, theta_data, w_data  #for memory

        #close outfile
        ncoutfile.close()

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

def get_vent_coeff(N_Re):
    """
    get ventilation coefficient (from pruppacher and klett p 541)
    """
    f = np.piecewise(N_Re, [N_Re < 2.46, N_Re >= 2.46], \
                    [lambda N_Re: 1. + 0.086*N_Re, \
                    lambda N_Re: 0.78 + 0.27*N_Re**0.5])
    return f

if __name__ == "__main__":
    main()
