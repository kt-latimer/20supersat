"""
Create and save file in DATA_DIR with secondary variables derived from raw WRF
output. Two files are created; one for BG, one for PI. Each file contains the
variables 'sumr_cloud' (mean cloud droplet radius; m), 'sumr_rain' (mean rain
drop radius; m).
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
g = 9.8 #grav accel (m/s^2)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
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
                        + 'wrfout_d01_sumr_vars', 'w')

        #make file dimensions
        ncoutfile.createDimension('west_east', 450)
        ncoutfile.createDimension('south_north', 450)
        ncoutfile.createDimension('bottom_top', 66)
        ncoutfile.createDimension('Time', 84)
        dims = ('Time', 'bottom_top', 'south_north', 'west_east')

        #get primary variables from wrf output
        PB = ncprimvars['PB'][...]
        P = ncprimvars['P'][...]
        T = ncprimvars['T'][...]
        theta_data = T_0 + T 
        pres_data = PB + P
        temp_data = theta_data*np.power((pres_data/P_0), R_a/C_ap)
        del PB, P, T, theta_data #for memory

        rho_air_data = pres_data/(R_a*temp_data)

        #compute mean radius of cloud droplets
        diams = [4*(2.**(i/3.))*10**(-6) for i in range(33)] #bin diams in m
        rads = [d/2. for d in diams]
        
        rcloud_wtsum = np.empty(np.shape(pres_data))
        rrain_wtsum = np.empty(np.shape(pres_data))
        Ncloud_sum = np.empty(np.shape(pres_data))
        Nrain_sum = np.empty(np.shape(pres_data))

        for i in range(33):
            r_i = rads[i]
            ff_i_wrf = ncprimvars['ff1i'+f'{i+1:02}'][...]
            N_i = ff_i_wrf/(4./3.*np.pi*r_i**3.*rho_w/rho_air_data)
            if i < 15:
                Ncloud_sum += N_i
                rcloud_wtsum += N_i*r_i
            else:
                Nrain_sum += N_i
                rrain_wtsum += N_i*r_i
            del ff_i_wrf, N_i #for memory

        sumrcloud_data = rcloud_wtsum
        sumrrain_data = rrain_wtsum 
        nconccloud_data = Ncloud_sum
        nconcrain_data = Nrain_sum

        #for memory
        del Ncloud_sum, Nrain_sum 

        #make variables for netCDF file
        sumrcloud = ncoutfile.createVariable('sumrcloud', np.dtype('float32'), dims)
        sumrrain = ncoutfile.createVariable('sumrrain', np.dtype('float32'), dims)
        nconccloud = ncoutfile.createVariable('nconccloud', np.dtype('float32'), dims)
        nconcrain = ncoutfile.createVariable('nconcrain', np.dtype('float32'), dims)
        
        #write data to variables
        sumrcloud[...] = sumrcloud_data
        sumrrain[...] = sumrrain_data
        nconccloud[...] = nconccloud_data
        nconcrain[...] = nconcrain_data

        del sumrcloud_data, sumrrain_data, nconccloud_data, nconcrain_data

        #close outfile
        ncoutfile.close()

if __name__ == "__main__":
    main()
