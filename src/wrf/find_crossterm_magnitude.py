"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import linregress, get_ss_qss, \
                        get_ss_qss_components, get_lwc

versionstr = 'v1_'

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

z_lim_1 = (1000, 1500)
z_lim_2 = (3000, 4000)
z_lim_3 = (1500, 2500)

lwc_cutoff_val = 1.e-4
w_cutoff_val = 1

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    for case_label in case_label_dict.keys():
        for z_lim in [z_lim_3]:# [z_lim_1, z_lim_2]:
            print(case_label, z_lim)
            calc_and_print_crossterm(case_label, \
                case_label_dict[case_label], cutoff_bins, \
                full_ss, incl_rain, incl_vent, z_lim)

def calc_and_print_crossterm(case_label, \
                case_dir_name, cutoff_bins, \
                full_ss, incl_rain, incl_vent, z_lim):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    A, B, meanr, nconc = get_ss_qss_components(met_vars, \
                            dsdsum_vars, cutoff_bins, full_ss, \
                            incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    wcu_filter_inds = np.logical_and.reduce((
                    (lwc > lwc_cutoff_val), \
                    (w > w_cutoff_val), \
                    (temp > 273)))
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_cutoff_val), \
                    (w > w_cutoff_val), \
                    (z > z_lim[0]), \
                    (z < z_lim[1]), \
                    (temp > 273)))

    ss_qss = ss_qss[filter_inds]
    A = A[filter_inds]
    B = B[filter_inds]
    meanr = meanr[filter_inds]
    nconc = nconc[filter_inds]
    w = w[filter_inds]

    print(np.shape(filter_inds))
    print(np.sum(wcu_filter_inds))
    print(np.sum(filter_inds))
    print(np.mean(ss_qss))
    print(np.mean(A*w/(4*np.pi*B))*np.mean(1./(meanr*nconc))*100.) #as a percentage
    print(np.mean(A*w/(4*np.pi*B))/np.mean(meanr*nconc)*100.) #as a percentage
    print(np.mean(A*w/(4*np.pi*B)))

if __name__ == "__main__":
    main()
