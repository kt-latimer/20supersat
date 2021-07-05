"""
Save data file with cumulative sums of volumes by bin
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from phys_consts import *
from wrf import BASE_DIR, DATA_DIR, FIG_DIR, WRF_bin_radii, n_WRF_bins
from wrf.dsd_data_functions import get_bin_nconc


case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')

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

    r3n_sum_data = np.zeros(np.shape(rho_air_data))

    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_drop_volume_cdf', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    for i in range(1, n_WRF_bins+1):
        nconc_i = get_bin_nconc(i, input_vars, rho_air_data)
        r_i = bin_radii[i-1]

        r3n_sum_data += nconc_i*r_i**3.

        var_name = 'r3n_sum_' + str(i)
        var_data = r3n_sum_data

        output_var = output_file.createVariable(var_name, np.dtype('float32'), dim_4d)
        output_var[:] = var_data

    input_file.close()
    output_file.close()

if __name__ == "__main__":
    main()
