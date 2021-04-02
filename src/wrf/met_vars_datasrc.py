"""
make meteorological data files
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR 
from wrf.met_data_functions import get_A, get_B, get_e_sat, get_LH, \
                                        get_LWC_cloud, get_LWC_rain, \
                                        get_nconc_cloud, get_nconc_rain, \
                                        get_pres, get_rain_rate, \
                                        get_rho_air, get_ss_wrf, \
                                        get_temp, get_w, get_x, get_y, get_z

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_3d = ('Time', 'south_north', 'west_east')
dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')
var_names = ['A', 'B', 'e_sat', 'LH', 'LWC_cloud', 'LWC_rain', \
             'nconc_cloud', 'nconc_rain', 'pres', 'rho_air', \
             'rain_rate', 'ss_wrf', 'temp', 'w', 'x', 'y', 'z']
var_dims = [dim_4d, dim_4d, dim_4d, dim_4d, dim_4d, dim_4d, \
            dim_4d, dim_4d, dim_4d, dim_4d, dim_4d, dim_4d, \
            dim_4d, dim_4d, dim_3d, dim_3d, dim_4d]
var_calculation_functions = [get_A, get_B, get_e_sat, get_LH, \
                             get_LWC_cloud, get_LWC_rain, \
                             get_nconc_cloud, get_nconc_rain, \
                             get_pres, get_rho_air, get_rain_rate, \
                             get_ss_wrf, get_temp, get_w, get_x, get_y, get_z]

def main():

    for case_label in case_label_dict.keys():
        make_met_file(case_label, case_label_dict[case_label])

def make_met_file(case_label, case_dir_name):

    #get input file variables
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables
    #input_file.close()

    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_met_vars', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    for i, var_name in enumerate(var_names):
        output_var = output_file.createVariable(var_name, \
                                    np.dtype('float32'), var_dims[i]) 
        output_var_data = var_calculation_functions[i](input_vars)
        output_var[:] = output_var_data

    input_file.close()
    output_file.close()

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

if __name__ == "__main__":
    main()
