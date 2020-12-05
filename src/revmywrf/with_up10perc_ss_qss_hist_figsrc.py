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
from revmywrf.ss_qss_calculations import get_ss, get_lwc

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

#change_dsd_corr = False
#cutoff_bins = False 
#incl_rain = False
#incl_vent = False
#full_ss = False

def main():
    
    versionnum = int(os.environ['versionnum'])
    (dummy_bool, cutoff_bins, full_ss, \
        incl_rain, incl_vent) = get_boolean_params(versionnum)
    versionstr = 'v' + str(versionnum) + '_'

    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_hist(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_qss_hist(case_label, case_dir_name, \
                                cutoff_bins, full_ss, incl_rain, \
                                incl_vent, versionstr):

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

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    del lwc, temp #for memory

    w_filt = w[filter_inds]
    up10perc_cutoff = np.percentile(w_filt, 90)
    up10perc_inds = np.logical_and.reduce((
                            (filter_inds), \
                            (w > up10perc_cutoff)))
    up10perc_ss_qss = ss_qss[up10perc_inds]

    ss_qss = ss_qss[filter_inds]

    print('mean all: ', np.nanmean(ss_qss))
    print('median all: ', np.nanmedian(ss_qss))
    print('stdev all: ', np.nanstd(ss_qss))
    print('max all: ' + str(np.nanmax(ss_qss)))
    print('# pts ss > 2% all: ' + str(np.sum(ss_qss > 2)))
    print('mean up10perc: ', np.nanmean(up10perc_ss_qss))
    print('median up10perc: ', np.nanmedian(up10perc_ss_qss))
    print('stdev up10perc: ', np.nanstd(up10perc_ss_qss))
    print('max up10perc: ' + str(np.nanmax(up10perc_ss_qss)))
    print('# pts ss > 2% up10perc: ' + str(np.sum(up10perc_ss_qss > 2)))
    print(case_label)
    print('# pts total: ' + str(np.sum(ss_qss < 200)))

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    #bins = [0+0.7*i for i in range(30)]
    n, bins, patches = ax.hist(ss_qss, bins=30, density=False, \
                            label='All data satisfying filtering criteria')
    ax.hist(up10perc_ss_qss, bins=bins, density=False, \
                histtype='stepfilled', linewidth=3, \
                edgecolor='r', facecolor=(0, 0, 0, 0), \
                label='Upper 10th percentile updrafts out of filtered data')
    #ax.hist(ss_qss, bins=bins, density=False)
    ax.set_xlabel('SS (%)')
    ax.set_ylabel('Count')
    ax.set_title(case_label + ' SS_QSS distb' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent) \
                    + ', full_ss=' + str(full_ss))
    plt.legend()
    outfile = FIG_DIR + versionstr + 'with_up10perc_ss_qss_hist_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

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
