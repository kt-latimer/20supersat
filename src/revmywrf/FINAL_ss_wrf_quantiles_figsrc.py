"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import numpy as np

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
versionstr = 'v1_'
colors_arr = cm.get_cmap('magma', 10).colors
magma_purple = colors_arr[3]
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
                            
lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    ss_qss_combined = None
    ss_wrf_combined = None

    for case_label in case_label_dict.keys():
        ss_qss, ss_wrf = get_ss_data(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)
        print(case_label)
        print(np.shape(ss_qss))
        print(np.shape(ss_wrf))

        ss_qss_combined = add_to_combined_array(ss_qss, ss_qss_combined)
        ss_wrf_combined = add_to_combined_array(ss_wrf, ss_wrf_combined)

    make_and_save_ss_wrf_quantiles(ss_qss_combined, \
                        ss_wrf_combined, versionstr)

def add_to_combined_array(ss, ss_combined):

    if ss_combined is None:
        return ss
    else:
        return np.concatenate((ss_combined, ss))

def get_ss_data(case_label, case_dir_name, \
                                cutoff_bins, full_ss, \
                                incl_rain, incl_vent):

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

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    return ss_qss, ss_wrf

def make_and_save_ss_wrf_quantiles(ss_qss_combined, \
                        ss_wrf_combined, versionstr):

    m, b, R, sig = linregress(ss_qss_combined, ss_wrf_combined)
    print(m, b, R**2.)
    ss_pred_combined = m*ss_qss_combined + b

    fig, ax = plt.subplots()

    quantiles = [5 + 10*i for i in range(10)]
    for quantile in quantiles:
        ax = plot_ss_wrf_quantile_curve(ax, quantile, \
                ss_pred_combined, ss_wrf_combined)
        
    ax.set_aspect('equal', 'box')

    ax.set_xlabel(r'$SS_{pred}$ (%)')
    ax.set_ylabel(r'$SS_{WRF}$ (%)')
    fig.suptitle('Distribution of $SS_{WRF}$ vs $SS_{pred}$ - WRF')

    outfile = FIG_DIR + versionstr + 'FINAL_ss_wrf_quantiles_figure.png'
    plt.savefig(outfile)
    plt.close()    

def plot_ss_wrf_quantile_curve(ax, quantile, \
            ss_pred_combined, ss_wrf_combined):

    min_ss_pred = np.min(ss_pred_combined)
    max_ss_pred = np.max(ss_pred_combined)

    ss_pred_bins = np.arange(min_ss_pred, max_ss_pred+d_ss, d_ss)

    ss_pred_vals = []
    ss_wrf_vals = []
    
    for i, lower_bin_edge in enumerate(ss_pred_bins[:-1]):
        upper_bin_edge = ss_pred_bins[i+1]
        bin_midpoint = 1./2.*(lower_bin_edge + upper_bin_edge)
        
        bin_filter_inds = np.logical_and.reduce(( 
                            (ss_pred_combined >= lower_bin_edge), \
                            (ss_pred_combined < upper_bin_edge)))

        if np.sum(bin_filter_inds) != 0:
            ss_wrf_bin = ss_wrf_combined[bin_filter_inds]
            ss_wrf_quantile = np.percentile(ss_wrf_vals, quantile)
        else:
            ss_wrf_quantile = None

        ss_pred_vals.append(bin_midpoint)
        ss_wrf_vals.append(ss_wrf_quantile)

    print(quantile)
    print(ss_pred_vals)
    print(ss_wrf_vals)
    print()
    ax.plot(ss_pred_vals, ss_wrf_vals, color=magma_purple)
    ax.text(ss_pred_vals[-1] + 2, ss_wrf_vals[-1], str(quantile)+r'$^{th}$')

    return ax

if __name__ == "__main__":
    main()
