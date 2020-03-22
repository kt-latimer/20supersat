"""
Create and save file in DATA_DIR with secondary variables derived from raw WRF
output. Two files are created; one for BG, one for PI. Each file contains the
variables 'meanr' (mean cloud drop radius; m), 'nconc' (cloud drop number
concentration; m^-3), 'pres' (total pressure; Pa), 'rho_air' (dry air density
from ideal gas law; kg/m^3), 'temp' (total temperature; K), 'theta' 
(total potential temperature; K), and 'w' (vertical wind velocity averaged to 
the centered (mass) grid; m/s).
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
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
        ncoutfile = Dataset(DATA_DIR + model_dir + 'wrfout_d01_secondary_vars', 'w')

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
        W = ncprimvars['W'][...]
        #vertical wind velocity is on a staggered grid; take NN average to
        #reshape to mass grid
        w_data = (W[:,0:-1,:,:] + W[:,1:,:,:])/2

        #calculate secondary variables
        theta_data = T_0 + T 
        pres_data = PB + P
        temp_data = theta_data*np.power((pres_data/P_0), R_a/C_ap)
        rho_air_data = pres_data/(R_a*temp_data) 

        #compute mean radius of cloud droplets
        diams = [4*(2.**(i/3.))*10**(-6) for i in range(15)] #bin diams in m
        rads = [d/2. for d in diams] 
        rN_sum = np.empty(np.shape(P))
        N_sum = np.empty(np.shape(P))
        for i in range(15):
            r = rads[i]
            ff_i_wrf = ncprimvars['ff1i'+f'{i+1:02}'][...]
            N_i = ff_i_wrf/(4./3.*np.pi*r**3.*rho_w/rho_air_data)
            N_sum += N_i
            rN_sum += N_i*r
        meanr_data = rN_sum/N_sum
        nconc_data = N_sum

        #make variables for netCDF file
        meanr = ncoutfile.createVariable('meanr', np.dtype('float32'), dims)
        nconc = ncoutfile.createVariable('nconc', np.dtype('float32'), dims)
        pres = ncoutfile.createVariable('pres' , np.dtype('float32'), dims)
        rho_air = ncoutfile.createVariable('rho_air', np.dtype('float32'), dims)
        temp = ncoutfile.createVariable('temp', np.dtype('float32'), dims)
        theta = ncoutfile.createVariable('theta', np.dtype('float32'), dims)
        w = ncoutfile.createVariable('w', np.dtype('float32'), dims)
        
        #write data to variables
        meanr[...] = meanr_data
        nconc[...] = nconc_data
        pres[...] = pres_data
        rho_air[...] = rho_air_data
        temp[...] = temp_data
        theta[...] = theta_data
        w[...] = w_data

        #close outfile
        ncoutfile.close()

if __name__ == "__main__":
    main()
