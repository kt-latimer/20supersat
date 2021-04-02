"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

#for plotting
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
    
    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_vs_ss_wrf(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)

def make_and_save_ss_qss_vs_ss_wrf(case_label, case_dir_name, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = met_vars['LWC_cloud'][...] + met_vars['LWC_rain'][...]

    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
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
    z = z[filter_inds]

    fig, ax = plt.subplots()
    
    ss_bins = get_ss_bins(ss_min, ss_max, d_ss)
    zs = get_zs(z, ss_bins, ss_qss, ss_wrf)

    im = ax.pcolormesh(ss_bins, ss_bins, zs, cmap=plt.cm.magma_r)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'z (m)')

    ax.set_aspect('equal', 'box')

    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'$SS_{WRF}$ (%)')

    outfile = FIG_DIR + 'z_heatmap_ss_qss_vs_ss_wrf_' \
                            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def get_zs(z, ss_bins, ss_qss, ss_wrf):

    n_bins = np.shape(ss_bins)[0] - 1
    zs = np.zeros((n_bins, n_bins))

    for i, val1 in enumerate(ss_bins[:-1]):
        for j, val2 in enumerate(ss_bins[:-1]):
            ss_qss_lo = val1
            ss_qss_hi = ss_bins[i+1]
            ss_wrf_lo = val2
            ss_wrf_hi = ss_bins[j+1]

            bin_inds = np.logical_and.reduce((( \
                            (ss_qss >= ss_qss_lo), \
                            (ss_qss < ss_qss_hi), \
                            (ss_wrf >= ss_wrf_lo), \
                            (ss_wrf < ss_wrf_hi))))

            if np.sum(bin_inds) > 0:
                z_filt = z[bin_inds]
                zs[i, j] = np.mean(z_filt)
            else:
                zs[i, j] = np.nan 

    return zs

def get_ss_bins(ss_min, ss_max, d_ss):

    ss_bins = np.arange(ss_min, ss_max, d_ss)

    return ss_bins

if __name__ == "__main__":
    main()
