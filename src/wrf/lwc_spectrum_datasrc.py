"""
calculating quantities from mid-radius sector WRF DSDs
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR 
from wrf.dsd_data_functions import get_bin_nconc

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

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')

###
### bin sizes and regime params
###
bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

def main():

    for case_label in case_label_dict.keys():
        make_lwc_spectrum_file(case_label, case_label_dict[case_label])

def make_lwc_spectrum_file(case_label, case_dir_name):

    print(case_label)

    #get met input file variables (naming a bit confusing here but trying
    #to maintain consistency with make_net_vars code...)
    met_input_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_input_vars = met_input_file.variables

    #get relevant environmental data 
    rho_air_data = met_input_vars['rho_air'][...]

    #close met input file for memory
    met_input_file.close()
    
    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    nconc_sum_data = np.zeros(np.shape(rho_air_data))
    r3n_sum_data = np.zeros(np.shape(rho_air_data))

    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_lwc_spectrum_vars', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    for i in range(1, 34):
        nconc_i = get_bin_nconc(i, input_vars, rho_air_data)
        r_i = bin_radii[i-1]

        nconc_sum_data += nconc_i
        r3n_sum_data += nconc_i*r_i**3.

        var_name = 'r3n_sum_' + str(i)
        var_data = r3n_sum_data

        output_var = output_file.createVariable(var_name, np.dtype('float32'), dim_4d)
        output_var[:] = var_data

    input_file.close()
    output_file.close()

if __name__ == "__main__":
    main()
