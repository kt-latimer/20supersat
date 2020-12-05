"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss
pres_min = 40000 
d_pres = 5000 
pres_max = 110000

def main():
    
    versionnum = int(os.environ['versionnum'])
    (dummy_bool, cutoff_bins, full_ss, \
        incl_rain, incl_vent) = get_boolean_params(versionnum)
    versionstr = 'v' + str(versionnum) + '_'

    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_vs_pres(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_qss_vs_pres(case_label, case_dir_name, \
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
    pres = met_vars['pres'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    ss_qss = ss_qss[filter_inds]
    pres = pres[filter_inds]

    #plot the supersaturations against each other with regression line
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ss_bins = get_bins(ss_min, ss_max, d_ss)
    pres_bins = get_bins(pres_min, pres_max, d_pres)

    h = ax.hist2d(ss_qss, pres, bins=[ss_bins, pres_bins], cmin=1, \
        norm=matplotlib.colors.LogNorm(), cmap=plt.cm.plasma)
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label(r'$N_{points}$')

    ax.set_xlim((ss_min, ss_max))
    ax.set_ylim((pres_min, pres_max))

    avg_ss_qss = get_avg_ss_qss(ss_qss, pres, pres_bins)
    print(avg_ss_qss)

    ax.scatter(avg_ss_qss, pres_bins[:-1], c='k', s=64)
    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'z (m)')
    outfile = FIG_DIR + versionstr + 'heatmap_ss_qss_vs_pres_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_bins(x_min, x_max, d_x):

    bins = np.arange(x_min, x_max, d_x)

    return bins

def get_avg_ss_qss(ss_qss, pres, pres_bins):

    avg_ss_qss = np.zeros(np.shape(pres_bins)[0] - 1)
    print(np.shape(avg_ss_qss))

    for i, val in enumerate(pres_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = pres_bins[i+1]

        bin_filter = np.logical_and.reduce((
                        (pres > lower_bin_edge), \
                        (pres < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_ss_qss[i] = np.nan
        else:
            ss_qss_pres_slice = ss_qss[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_pres_slice)

    return avg_ss_qss
        
def get_boolean_params(versionnum):

    versionnum = versionnum - 1 #for modular arithmetic
    
    if versionnum > 23:
        return versionnum

    #here this is a dummy var but keeping for consistency with
    #revhalo for now aka for my sanity
    if versionnum < 12:
        change_cas_corr = False
    else:
        change_cas_corr = True

    if versionnum % 12 < 6:
        cutoff_bins = False
    else:
        cutoff_bins = True

    if versionnum % 6 < 3:
        full_ss = False
    else:
        full_ss = True

    if versionnum % 3 == 0:
        incl_rain = False
    else:
        incl_rain = True

    if versionnum % 3 == 2:
        incl_vent = True
    else:
        incl_vent = False

    return (change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

if __name__ == "__main__":
    main()
