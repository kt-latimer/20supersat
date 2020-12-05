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
z_min = -100
d_z = 250
z_max = 8000

def main():
    
    versionnum = int(os.environ['versionnum'])
    (dummy_bool, cutoff_bins, full_ss, \
        incl_rain, incl_vent) = get_boolean_params(versionnum)
    versionstr = 'v' + str(versionnum) + '_'

    for case_label in case_label_dict.keys():
        if case_label == 'Unpolluted':
            continue
        make_and_save_ss_qss_vs_z(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_qss_vs_z(case_label, case_dir_name, \
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
    z = met_vars['z'][...]
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
    z = z[filter_inds]

    #plot the supersaturations against each other with regression line
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ss_bins = get_bins(ss_min, ss_max, d_ss)
    z_bins = get_bins(z_min, z_max, d_z)

    h = ax.hist2d(ss_qss, z, bins=[ss_bins, z_bins], cmin=1, \
        norm=matplotlib.colors.LogNorm(), cmap=plt.cm.plasma)
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label(r'$N_{points}$')

    ax.set_xlim((ss_min, ss_max))
    ax.set_ylim((z_min, z_max))

    avg_ss_qss = get_avg_ss_qss(ss_qss, z, z_bins)
    print(avg_ss_qss)

    ax.scatter(avg_ss_qss, z_bins[:-1], c='k', s=64)
    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'z (m)')
    outfile = FIG_DIR + versionstr + 'heatmap_ss_qss_vs_z_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_bins(x_min, x_max, d_x):

    bins = np.arange(x_min, x_max, d_x)

    return bins

def get_avg_ss_qss(ss_qss, z, z_bins):

    avg_ss_qss = np.zeros(np.shape(z_bins)[0] - 1)
    print(np.shape(avg_ss_qss))

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        bin_filter = np.logical_and.reduce((
                        (z > lower_bin_edge), \
                        (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_ss_qss[i] = np.nan
        else:
            ss_qss_z_slice = ss_qss[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_z_slice)

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

def print_point_count_per_quadrant(ss_qss, ss_wrf):

    n_q1 = np.sum(np.logical_and(ss_qss > 0, ss_wrf > 0))
    n_q2 = np.sum(np.logical_and(ss_qss < 0, ss_wrf > 0))
    n_q3 = np.sum(np.logical_and(ss_qss < 0, ss_wrf < 0))
    n_q4 = np.sum(np.logical_and(ss_qss > 0, ss_wrf < 0))
    
    print('Number of points in Q1:', n_q1)
    print('Number of points in Q2:', n_q2)
    print('Number of points in Q3:', n_q3)
    print('Number of points in Q4:', n_q4)
    print()
    print('Domain:', np.min(ss_qss), np.max(ss_qss))
    print('Range:', np.min(ss_wrf), np.max(ss_wrf))

if __name__ == "__main__":
    main()
