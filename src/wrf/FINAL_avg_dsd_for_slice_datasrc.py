"""
Save data file with avg value of (vent-corrected) nconc in WCUs for each size
bin, within given altitude slice
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from phys_consts import *
from wrf import BASE_DIR, DATA_DIR, FIG_DIR, n_WRF_bins
from wrf.dsd_data_functions import get_bin_nconc, get_bin_vent_coeff
from wrf.met_data_functions import get_dyn_visc
from wrf.ss_functions import get_lwc
            
lwc_cutoff = 1.e-4
w_cutoff = 1

#altitude slice boundaries (m)
slice_lo_edge = 1500
slice_hi_edge = 2500

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = False 
incl_rain = True 
incl_vent = True 
full_ss = True 

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_meanr_dicts(case_label, case_label_dict[case_label])

def make_and_save_meanr_dicts(case_label, case_dir_name):

    print(case_label)

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    pres = met_vars['pres'][...]
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #get values for calculating ventilation coefficient
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air**2./(eta**4.*g*rho_w) #pr&kl p 418

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff), \
                            (temp > 273), \
                            (w > w_cutoff), \
                            (z > slice_lo_edge), \
                            (z < slice_up_edge)))

    avg_nconc_by_bin = np.zeros(n_WRF_bins)
    avg_nconc_by_bin_vent_corr = np.zeros(n_WRF_bins)

    for i in range(1, n_WRF_bins+1):
        nconc_i = get_bin_nconc(i, input_vars, rho_air)
        f_i = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                                    pres, rho_air, temp)
        nconc_i = nconc_i[filter_inds]
        f_i = f_i[filter_inds]
        avg_nconc_by_bin[i-1] = np.mean(nconc_i)
        avg_nconc_by_bin_vent_corr[i-1] = np.mean(nconc_i*f_i)

    #close files for memory
    input_file.close()

    nconc_by_bin_dict = {'no_vent': avg_nconc_by_bin, \
                         'with_vent': avg_nconc_by_bin_vent_corr}

    nconc_by_bin_filename = DATA_DIR + 'avg_dsd_for_slice_' \
                                    + case_label + '_data.npy'

    np.save(nconc_by_bin_filename, nconc_by_bin_dict)

if __name__ == "__main__":
    main()
