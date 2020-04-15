"""
Add SS_WRF, LWC, and latent heating (both K/s and J/(m^3 s)) to secondary_vars
files (in first run, created secondary_wrf_vars_with_ss_v3 with lwc, meanr, and 
nconc only from clouds; in second run, created secondary_wrf_vars_with_ss_v4 
with those things from both clouds and rain (also with lwc separated into
cloud, rain, and tot). 
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
        ncinfileprim = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncinfileprim.variables

        #load primary datafiles
        ncinfilesec = Dataset(DATA_DIR + model_dir \
                        + 'wrfout_d01_secondary_vars_with_ss_v3', 'r')
        ncsecvars = ncinfilesec.variables
        #dims = ncinfilesec.dimensions
        
        #get primary variables from wrf output
        lh_K_s_data = ncprimvars['TEMPDIFFL'][...]
        lwc_cloud_data = ncprimvars['QCLOUD'][...]
        lwc_rain_data = ncprimvars['QRAIN'][...]
        ss_wrf_data = ncprimvars['SSW'][...]

        #make variables for netCDF file
        meanr_data = ncsecvars['meanr'][...]
        nconc_data = ncsecvars['nconc'][...]
        pres_data = ncsecvars['pres'][...]
        rho_air_data = ncsecvars['rho_air'][...]
        ss_qss_data = ncsecvars['ss_qss'][...]
        ss_ry_data = ncsecvars['ss_ry'][...]
        temp_data = ncsecvars['temp'][...]
        theta_data = ncsecvars['theta'][...]
        w_data = ncsecvars['w'][...]

        #latent heating in J/(m^3 s)
        lh_J_m3_s_data = lh_K_s_data*C_ap*rho_air_data
       
        ncinfileprim.close()
        #create secondary datafile
        ncoutfile = Dataset(DATA_DIR + model_dir \
                        + 'wrfout_d01_secondary_vars_with_ss_v4', 'w')

        #make file dimensions
        ncoutfile.createDimension('west_east', 450)
        ncoutfile.createDimension('south_north', 450)
        ncoutfile.createDimension('bottom_top', 66)
        ncoutfile.createDimension('Time', 84)
        dims = ('Time', 'bottom_top', 'south_north', 'west_east')
        #dims = ncoutfile.dimensions 

        lh_J_m3_s = ncoutfile.createVariable('lh_J_m3_s', np.dtype('float32'), dims)
        lh_K_s = ncoutfile.createVariable('lh_K_s', np.dtype('float32'), dims)
        lwc_cloud = ncoutfile.createVariable('lwc_cloud', np.dtype('float32'), dims)
        lwc_rain = ncoutfile.createVariable('lwc_rain', np.dtype('float32'), dims)
        lwc_tot = ncoutfile.createVariable('lwc_tot', np.dtype('float32'), dims)
        ss_wrf = ncoutfile.createVariable('ss_wrf', np.dtype('float32'), dims)

        meanr = ncoutfile.createVariable('meanr', np.dtype('float32'), dims)
        nconc = ncoutfile.createVariable('nconc', np.dtype('float32'), dims)
        pres = ncoutfile.createVariable('pres' , np.dtype('float32'), dims)
        rho_air = ncoutfile.createVariable('rho_air', np.dtype('float32'), dims)
        ss_qss = ncoutfile.createVariable('ss_qss', np.dtype('float32'), dims)
        ss_ry = ncoutfile.createVariable('ss_ry', np.dtype('float32'), dims)
        temp = ncoutfile.createVariable('temp', np.dtype('float32'), dims)
        theta = ncoutfile.createVariable('theta', np.dtype('float32'), dims)
        w = ncoutfile.createVariable('w', np.dtype('float32'), dims)

        #write data to variables
        lh_J_m3_s[...] = lh_J_m3_s_data
        lh_K_s[...] = lh_K_s_data
        lwc_cloud[...] = lwc_cloud_data
        lwc_rain[...] = lwc_rain_data
        lwc_tot[...] = lwc_cloud_data + lwc_rain_data
        ss_wrf[...] = ss_wrf_data

        meanr[...] = meanr_data
        nconc[...] = nconc_data
        pres[...] = pres_data
        rho_air[...] = rho_air_data
        ss_qss[...] = ss_qss_data
        ss_ry[...] = ss_ry_data
        temp[...] = temp_data
        theta[...] = theta_data
        w[...] = w_data

        #close outfile
        ncoutfile.close()

if __name__ == "__main__":
    main()
