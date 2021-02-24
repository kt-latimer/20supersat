"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
versionstr = 'v3_'
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
                            
lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

d_ss_lg = 2 
d_ss_sm = 0.1 

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_ss_wrf_distbs(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_wrf_distbs(case_label, case_dir_name, \
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

    fig, ax = plt.subplots()
    
    ss_qss_min = np.nanmin(ss_qss)
    ss_qss_max = np.nanmax(ss_qss)
    ss_wrf_min = np.nanmin(ss_wrf)
    ss_wrf_max = np.nanmax(ss_wrf)

    ss_qss_bins = get_ss_bins(ss_qss_min, ss_qss_max, d_ss_lg)
    colors_arr = cm.get_cmap('magma', np.shape(ss_qss_bins)[0]-1).colors

    ##do all the ss_qss range bars first so they're in the background
    #for i, color in enumerate(colors_arr):
    #    ax.axvspan(ss_qss_bins[i], ss_qss_bins[i+1], \
    #                alpha=0.5, color=color, edgecolor='white')

    max_dn_vals = []
    max_ss_vals = []
    bin_mid_vals = []

    for i, color in enumerate(colors_arr):
        ax, max_dn, max_ss = add_distb_to_ax(ax, ss_qss_bins[i], ss_qss_bins[i+1], \
                                color, ss_qss, ss_wrf)
        max_dn_vals.append(max_dn)
        max_ss_vals.append(max_ss)
        bin_mid_vals.append(0.5*(ss_qss_bins[i] + ss_qss_bins[i+1]))

    ymin, ymax = ax.get_ylim()

    for i, color in enumerate(colors_arr):
        ax.plot([max_ss_vals[i], bin_mid_vals[i]], \
                [max_dn_vals[i], ymax*0.99], color='black', \
                marker='o', linestyle='--')
        ax.plot([ss_qss_bins[i]+d_ss_sm, ss_qss_bins[i+1]-d_ss_sm], \
                [ymax*0.99, ymax*0.99], color=color, linewidth=6, alpha=0.7)

    secax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    secax.set_xlabel(r'$SS_{QSS}$ (%)')

    ax.set_xlabel(r'$SS_{WRF}$ (%)')
    ax.set_ylabel(r'$\frac{dn_{points}}{dSS_{WRF}}$ (%$^{-1}$)')
    #ax.set_ylabel(r'$\frac{dN_{points}}{dSS_{WRF}}$ (%$^{-1}$)')
    ax.set_title(r'Distributions of $SS_{WRF}$ given $SS_{QSS}$' \
                    ' - WRF ' + case_label)

    outfile = FIG_DIR + versionstr + 'FINAL_ss_wrf_distbs_' \
            + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

def get_ss_bins(ss_min, ss_max, d_ss):

    ss_bins = np.arange(ss_min, ss_max+d_ss, d_ss)

    return ss_bins

def add_distb_to_ax(ax, lo_ss_bin, hi_ss_bin, color, ss_qss, ss_wrf):

    bin_filter_inds = np.logical_and.reduce((
                        (ss_qss >= lo_ss_bin), \
                        (ss_qss < hi_ss_bin)))

    if np.sum(bin_filter_inds) != 0:
        ss_wrf_vals = ss_wrf[bin_filter_inds]
        ss_wrf_vals_min = np.nanmin(ss_wrf_vals)
        ss_wrf_vals_max = np.nanmax(ss_wrf_vals)
        
        ss_wrf_bins = get_ss_bins(ss_wrf_vals_min, ss_wrf_vals_max, d_ss_sm)

        if np.shape(ss_wrf_bins)[0] > 0:
            dn, bins, patches = ax.hist(ss_wrf_vals, bins=ss_wrf_bins, \
                    facecolor=(0, 0, 0, 0.0), edgecolor=color, \
                    #histtype='stepfilled', density=False)
                    histtype='stepfilled', density=True)
            max_dn, max_ss = get_max_dn_and_ss(dn, bins) 
        else:
            max_dn = None
            max_ss = None
    else:
        max_dn = None
        max_ss = None

    return ax, max_dn, max_ss 

def get_max_dn_and_ss(dn, bins):

    max_dn = 0
    max_ss = 0

    for i, val in enumerate(dn):
        if val > max_dn:
            max_dn = val
            max_ss = 0.5*(bins[i] + bins[i+1])

    return max_dn, max_ss

if __name__ == "__main__":
    main()
