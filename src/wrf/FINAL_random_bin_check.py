"""
check d(N*r)/dlogDp for a random WRF bin in both simulation cases
"""

from netCDF4 import Dataset, MFDataset
import numpy as np

from phys_consts import *
from wrf import BASE_DIR, DATA_DIR, FIG_DIR, WRF_bin_radii, n_WRF_bins
from wrf.dsd_data_functions import get_bin_nconc, get_bin_vent_coeff
from wrf.met_data_functions import get_dyn_visc
from wrf.ss_functions import get_lwc

lwc_cutoff_val = 1.e-4
w_cutoff_val = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

z_lims = [(1000, 1500), (3000, 4000)]

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

WRF_bin_dlogDp = np.array([np.log10(2.**(1./3.)) for i in range(n_WRF_bins)])

def main():

    #hand-coded range to capture non-negligible concentration 
    random_bin_ind = np.random.randint(3, 17)
    for case_label in case_label_dict.keys():
        for z_lim in z_lims:
            bin_nconc = get_bin_nconc_slice(random_bin_ind, case_label, z_lim)
            print(case_label, z_lim, random_bin_ind, \
                WRF_bin_radii[random_bin_ind], \
                np.nanmean(bin_nconc)*WRF_bin_radii[random_bin_ind]/WRF_bin_dlogDp[random_bin_ind])

def get_bin_nconc_slice(random_bin_ind, case_label, z_lim):

    case_dir_name = case_label_dict[case_label]

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
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (w > w_cutoff_val), \
                            (z > z_lim[0]), \
                            (z < z_lim[1])))

    nconc_i = get_bin_nconc(random_bin_ind+1, input_vars, rho_air)
    f_i = get_bin_vent_coeff(random_bin_ind+1, eta, N_Be_div_r3, N_Bo_div_r2, \
                                                        N_P, pres, rho_air, temp)

    #close files for memory
    input_file.close()

    return nconc_i[filter_inds]*f_i[filter_inds] 

if __name__=="__main__":
    main()