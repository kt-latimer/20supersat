"""
calculating quantities from high-radius sector WRF DSDs
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from phys_consts import *
from wrf import BASE_DIR, DATA_DIR, FIG_DIR, WRF_bin_radii
from wrf.dsd_data_functions import get_bin_nconc, get_bin_vent_coeff
from wrf.met_data_functions import get_dyn_visc

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

###
### bin sizes and regime params
###
lower_cutoff = 102e-6
upper_cutoff = 100 #ie infinity

def main():

    for case_label in case_label_dict.keys():
        make_dsdsum_file(case_label, case_label_dict[case_label])

def make_dsdsum_file(case_label, case_dir_name):

    #get met input file variables (naming a bit confusing here but trying
    #to maintain consistency with make_net_vars code...)
    met_input_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_input_vars = met_input_file.variables

    #get relevant environmental data 
    pres_data = met_input_vars['pres'][...]
    rho_air_data = met_input_vars['rho_air'][...]
    temp_data = met_input_vars['temp'][...]

    #close met input file for memory
    met_input_file.close()

    #get values for calculating ventilation coefficient
    eta = get_dyn_visc(temp_data)
    sigma = sum([sigma_coeffs[i]*(temp_data - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air_data*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air_data**2./(eta**4.*g*rho_w) #pr&kl p 418
    
    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    nconc_sum_data = np.zeros(np.shape(temp_data))
    rn_sum_data = np.zeros(np.shape(temp_data))
    frn_sum_data = np.zeros(np.shape(temp_data))
    r3n_sum_data = np.zeros(np.shape(temp_data))
    fr3n_sum_data = np.zeros(np.shape(temp_data))

    for i in range(1, 33):
        r_i = bin_radii[i-1]
        #just doing nested loop here bc turning into a new function seems
        #like it will be even messier
        if r_i >= lower_cutoff and r_i < upper_cutoff:
            nconc_i = get_bin_nconc(i, input_vars, rho_air_data)
            f_i = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                            pres_data, rho_air_data, temp_data)
            nconc_sum_data += nconc_i
            rn_sum_data += nconc_i*r_i
            frn_sum_data += nconc_i*r_i*f_i
            r3n_sum_data += nconc_i*r_i**3.
            fr3n_sum_data += nconc_i*r_i**3.*f_i

    input_file.close()

    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_hi_dsdsum_vars', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    var_names = ['nconc_sum', 'rn_sum', 'r3n_sum', 'frn_sum', 'fr3n_sum']
    var_data = [nconc_sum_data, rn_sum_data, r3n_sum_data, \
                frn_sum_data, fr3n_sum_data]

    for i, var_name in enumerate(var_names):
        output_var = output_file.createVariable(var_name, np.dtype('float32'), dim_4d)
        output_var[:] = var_data[i]
        
    output_file.close()

if __name__ == "__main__":
    main()
