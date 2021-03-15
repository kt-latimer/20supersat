"""
combine dsd sum data files
"""
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR 

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')

def main():

    for case_label in case_label_dict.keys():
        combine_dsdsum_files(case_label, case_label_dict[case_label])

def combine_dsdsum_files(case_label, case_dir_name):

    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_all_dsdsum_vars_v2', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    section_strs = ['lo', 'mid', 'hi']

    for section_str in section_strs:
        add_section_vars_to_combined_file(case_dir_name, \
                                            section_str, output_file)

    output_file.close()

def add_section_vars_to_combined_file(case_dir_name, section_str, output_file):

    #get input file variables
    input_file = Dataset(DATA_DIR + case_dir_name + 'wrfout_d01_' +
                            section_str + '_dsdsum_vars_v2', 'r')
    input_vars = input_file.variables

    base_var_names = ['nconc_sum', 'rn_sum', 'r3n_sum', 'frn_sum', 'fr3n_sum']
    var_names = [var_name + '_' + section_str for var_name in base_var_names]

    for i, var_name in enumerate(var_names):
        output_var = output_file.createVariable(var_name, \
                                    np.dtype('float32'), dim_4d) 
        output_var_data = input_vars[base_var_names[i]][...]
        output_var[:] = output_var_data

if __name__ == "__main__":
    main()
