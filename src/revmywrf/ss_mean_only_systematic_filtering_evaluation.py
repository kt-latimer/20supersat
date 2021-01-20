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
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
versionstr = 'v1_'
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

data_dict = {'Polluted': {'ss_qss_mean_arr': None, 'ss_wrf_mean_arr': None}, 
             'Unpolluted': {'ss_qss_mean_arr': None, 'ss_wrf_mean_arr': None}}

def main():
    
    for case_label in case_label_dict.keys():
        ss_qss_mean_arr, ss_wrf_mean_arr = \
            get_filter_dependent_values(case_label, \
                case_label_dict[case_label], \
                cutoff_bins, full_ss, incl_rain, \
                incl_vent, versionstr)
        make_and_save_ss_mean_heatmap(ss_qss_mean_arr, \
                            ss_wrf_mean_arr, case_label)
        
        data_dict[case_label]['ss_qss_mean_arr'] = ss_qss_mean_arr 
        data_dict[case_label]['ss_wrf_mean_arr'] = ss_wrf_mean_arr
        
    filename = versionstr + 'ss_mean_only_systematic_filtering_evaluation_data.npy'
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
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100
    shape = np.shape(lwc)
    ntot = shape[0]*shape[1]*shape[2]*shape[3]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    ss_wrf_mean_arr = np.zeros((n_lwc_vals, n_w_vals)) 
    ss_qss_mean_arr = np.zeros((n_lwc_vals, n_w_vals)) 

    for i, j in product(range(n_lwc_vals), range(n_w_vals)):
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        
        #apply filtering criteria
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_filter_val), \
                        (temp > 273)))

        if np.sum(filter_inds) == 0:
            ss_qss_mean_arr[i, j] = np.nan 
            ss_wrf_mean_arr[i, j] = np.nan

        else:
            ss_qss_filt = ss_qss[filter_inds]
            ss_wrf_filt = ss_wrf[filter_inds]

            ss_qss_mean = np.mean(ss_qss_filt) 
            ss_wrf_mean = np.mean(ss_wrf_filt) 

            ss_qss_mean_arr[i, j] = ss_qss_mean 
            ss_wrf_mean_arr[i, j] = ss_wrf_mean 

    return ss_qss_mean_arr, ss_wrf_mean_arr

def make_and_save_ss_mean_heatmap(ss_qss_mean_arr, ss_wrf_mean_arr, case_label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(30, 15)

    im1 = ax1.imshow(ss_qss_mean_arr, cmap=rev_magma)
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel(r'$SS_{QSS}$ (%)')

    ax1.set_xticks(np.arange(n_lwc_vals))
    ax1.set_yticks(np.arange(n_w_vals))
    ax1.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax1.set_yticklabels(np.around(w_filter_vals, 2))

    ax1.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax1.set_ylabel('Min w cutoff (m/s)')

    im2 = ax2.imshow(ss_qss_mean_arr, cmap=magma)
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel(r'$SS_{WRF}$ (%)')

    ax2.set_xticks(np.arange(n_lwc_vals))
    ax2.set_yticks(np.arange(n_w_vals))
    ax2.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax2.set_yticklabels(np.around(w_filter_vals, 2))

    ax2.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    outfile = FIG_DIR + versionstr + 'ss_mean_heatmap_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
