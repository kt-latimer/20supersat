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

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_ss_qss, linregress

#for plotting
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

#modified color structure for histograms
magma_discrete = cm.get_cmap('magma', n_lwc_vals*n_w_vals)
colors = magma_discrete.colors 

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

data_dict = {'Polluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}, 
             'Unpolluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}}

def main():
    
    for case_label in case_label_dict.keys():
        case_data_dict = get_filter_dependent_values(case_label, \
                        case_label_dict[case_label], cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
        
        data_dict[case_label] = case_data_dict
        
    filename ='filtering_criteria_data.npy'
    np.save(DATA_DIR + filename, data_dict)

def get_filter_dependent_values(case_label, case_dir_name, \
                cutoff_bins, full_ss, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lh = met_vars['LH'][...]
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100
    shape = np.shape(lwc)
    ntot = shape[0]*shape[1]*shape[2]*shape[3]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    b_arr = np.zeros((n_lwc_vals, n_w_vals))
    lh_sum_arr = np.zeros((n_lwc_vals, n_w_vals))
    lwc_cutoffs_arr = np.zeros((n_lwc_vals, n_w_vals))
    m_arr = np.zeros((n_lwc_vals, n_w_vals))
    npts_arr = np.zeros((n_lwc_vals, n_w_vals))
    rsq_arr = np.zeros((n_lwc_vals, n_w_vals))
    w_cutoffs_arr = np.zeros((n_lwc_vals, n_w_vals))

    for i, j in product(range(n_lwc_vals), range(n_w_vals)):
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        lwc_cutoffs_arr[i][j] = lwc_filter_val
        w_cutoffs_arr[i][j] = w_filter_val
        
        #apply filtering criteria
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_filter_val), \
                        (temp > 273)))

        if np.sum(filter_inds) == 0:
            b_arr[i, j] = np.nan
            lh_sum_arr[i, j] = np.nan
            m_arr[i, j] = np.nan
            npts_arr[i, j] = np.nan
            rsq_arr[i, j] = np.nan

        else:
            lh_filt = lh[filter_inds]
            ss_qss_filt = ss_qss[filter_inds]
            ss_wrf_filt = ss_wrf[filter_inds]

            m, b, R, sig = linregress(ss_qss_filt, ss_wrf_filt)
            npts = np.shape(ss_qss_filt)[0]

            b_arr[i, j] = b
            lh_sum_arr[i, j] = np.sum(lh_filt)
            m_arr[i, j] = m
            npts_arr[i, j] = npts/ntot
            rsq_arr[i, j] = R**2.

    case_data_dir = {'b_arr': b_arr, 'lh_sum_arr': lh_sum_arr, \
            'lwc_cutoffs_arr': lwc_cutoffs_arr, 'm_arr': m_arr, \
            'npts_arr': npts_arr, 'rsq_arr': rsq_arr, \
            'w_cutoffs_arr': w_cutoffs_arr}

    return case_data_dir 

if __name__ == "__main__":
    main()
