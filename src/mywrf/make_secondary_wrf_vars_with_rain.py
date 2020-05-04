"""
Create and save file in DATA_DIR with secondary variables derived from raw WRF
output. Two files are created; one for BG, one for PI. Each file contains the
variables 'lh_J_m3_s' (latent heating by water vapor diffusion; J/(m^3 s)),
'lh_K_s' (latent heating by water vapor diffusion; K/s), 'lwc_cloud' (liquid 
water content in cloud droplets; g/g), 'lwc_rain' (lwc in rain drops; g/g), 
'lwc_tot' (lwc in cloud and rain; g/g), 'meanr' (mean cloud and rain drop 
radius; m), 'nconc' (cloud drop number concentration; m^-3), 'pres' (total 
pressure; Pa), 'rho_air' (dry air density from ideal gas law; kg/m^3), 'ss_qss'
(quasi-steady-state supersaturation, no ventilation), 'ss_ry' (supersaturation
from condensation rate as given in Rogers and Yau, no ventilation), 'ss_wrf' 
(supersaturation from WRF output), 'temp' (total temperature; K), 'theta' 
(total potential temperature; K), and 'w' (vertical wind velocity averaged to
the centered (mass) grid; m/s). 
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
                        + 'wrfout_d01_secondary_vars_with_rain', 'w')

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

        #compute mean radius of cloud droplets
        diams = [4*(2.**(i/3.))*10**(-6) for i in range(33)] #bin diams in m
        rads = [d/2. for d in diams] 
        rN_sum = np.empty(np.shape(pres_data))
        N_sum = np.empty(np.shape(pres_data))
        for i in range(33):
            r = rads[i]
            ff_i_wrf = ncprimvars['ff1i'+f'{i+1:02}'][...]
            N_i = ff_i_wrf/(4./3.*np.pi*r**3.*rho_w/rho_air_data)
            N_sum += N_i
            rN_sum += N_i*r
            del ff_i_wrf, N_i #for memory
        meanr_data = rN_sum/N_sum
        nconc_data = N_sum
        del N_sum, rN_sum #for memory
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
        meanr = ncoutfile.createVariable('meanr', np.dtype('float32'), dims)
        nconc = ncoutfile.createVariable('nconc', np.dtype('float32'), dims)
        pres = ncoutfile.createVariable('pres' , np.dtype('float32'), dims)
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
        meanr[...] = meanr_data
        nconc[...] = nconc_data
        pres[...] = pres_data
        rho_air[...] = rho_air_data
        ss_qss[...] = ss_qss_data
        ss_ry[...] = ss_ry_data
        ss_wrf[...] = ss_wrf_data
        temp[...] = temp_data
        theta[...] = theta_data
        w[...] = w_data

        del lh_K_s_data, lwc_cloud_data, lwc_rain_data, meanr_data, \
            nconc_data, pres_data, rho_air_data, ss_qss_data, ss_ry_data, \
            ss_wrf_data, temp_data, theta_data, w_data  #for memory

        #close outfile
        ncoutfile.close()

if __name__ == "__main__":
    main()
