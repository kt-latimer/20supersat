"""
1) make heatmap showing least squares slope and r-squared of ss_wrf vs ss_qss
for different vals of w and LWC cutoffs; 2) save ss_wrf and ss_qss
distributions (histograms) in the same parameter space
"""
from itertools import product
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
magma = cm.get_cmap('magma')
rev_magma = cm.get_cmap('magma_r')
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

log_lwc_min = -6
log_lwc_max = -2
n_lwc_vals = 9
lwc_filter_vals = 10**np.linspace(log_lwc_min, log_lwc_max, n_lwc_vals)

w_min = 0
w_max = 6 
n_w_vals = 7
w_filter_vals = np.linspace(w_min, w_max, n_w_vals)

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

data_dict = {'Polluted': {'lh_filt_arr': None, 'lh_tot_arr': None}, 
             'Unpolluted': {'lh_filt_arr': None, 'lh_tot_arr': None}}

def main():
    
    for case_label in case_label_dict.keys():
        lh_filt_arr, lh_tot_arr = \
            get_filter_dependent_values(case_label, \
                case_label_dict[case_label], \
                cutoff_bins, full_ss, incl_rain, \
                incl_vent, versionstr)
        make_and_save_lh_frac_heatmap(lh_filt_arr, lh_tot_arr, case_label)
        
        data_dict[case_label]['lh_filt_arr'] = lh_filt_arr 
        data_dict[case_label]['lh_tot_arr'] = lh_tot_arr
        
    filename = versionstr + 'lh_frac_only_systematic_filtering_evaluation_data.npy'
    np.save(DATA_DIR + filename, data_dict)

def get_filter_dependent_values(case_label, case_dir_name, \
                                cutoff_bins, full_ss, \
                                incl_rain, incl_vent, versionstr):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lh = met_vars['LH'][...] 
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    filter_inds = np.logical_and.reduce((
                    (lh > 0), \
                    (temp > 273)))
    lh_sum = np.sum(lh[filter_inds])

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    lh_filt_arr = np.zeros((n_lwc_vals, n_w_vals)) 
    lh_tot_arr = np.zeros((n_lwc_vals, n_w_vals)) #all entries the same

    for i, j in product(range(n_lwc_vals), range(n_w_vals)):
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        
        #apply filtering criteria
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_filter_val), \
                        (temp > 273)))

        if np.sum(filter_inds) == 0:
            lh_filt_arr[i, j] = np.nan
            lh_tot_arr[i, j] = lh_sum

        else:
            lh_filt_arr[i, j] = np.sum(lh[filter_inds]) 
            lh_tot_arr[i, j] = lh_sum

    return lh_filt_arr, lh_tot_arr 

def make_and_save_lh_frac_heatmap(lh_filt_arr, lh_tot_arr, case_label):

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)

    im = ax.imshow(lh_filt_arr.T/lh_tot_arr.T, cmap=rev_magma)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('LH fraction')

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    outfile = FIG_DIR + versionstr + 'lh_frac_heatmap_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
