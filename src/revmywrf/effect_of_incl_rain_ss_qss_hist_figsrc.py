"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_ss, get_lwc

#for plotting
versionstr = 'v1_'
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
    
    versionnums = [10, 12]
    boolean_param_tuples = [get_boolean_params(n) for n in versionnums]

    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_hist(case_label, \
                case_label_dict[case_label], boolean_param_tuples)

def make_and_save_ss_qss_hist(case_label, case_dir_name, \
                                boolean_param_tuples):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    ss_qss_arrays = []
    for boolean_param_tuple in boolean_param_tuples:
        ss_qss_arrays.append( \
            get_ss_qss_array_from_boolean_params(boolean_param_tuple, \
                                                    met_vars, \
                                                    dsdsum_vars))

    ss_qss_no_rain = ss_qss_arrays[0]
    ss_qss_incl_rain = ss_qss_arrays[1]
        
    #close files for memory
    met_file.close()
    dsdsum_file.close()

    print(case_label)

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)

    (n, no_rain_bins, patches) = ax.hist(ss_qss_no_rain, \
                                         bins=30, \
                                         alpha=0.6, \
                                         label='No rain', \
                                         density=False)
    ax.hist(ss_qss_incl_rain, \
             bins=no_rain_bins, \
             alpha=0.6, \
             label='With rain', \
             density=False)
    ax.set_xlabel('SS (%)')
    ax.set_ylabel('Count')
    ax.set_title(case_label + ' SS_QSS distb')
    fig.legend()
    outfile = FIG_DIR + versionstr + 'effect_of_incl_rain_ss_qss_hist_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_ss_qss_array_from_boolean_params(boolean_param_tuple, \
                                                    met_vars, \
                                                    dsdsum_vars):

    cutoff_bins = boolean_param_tuple[1]
    full_ss = boolean_param_tuple[2]
    incl_rain = boolean_param_tuple[3]
    incl_vent = boolean_param_tuple[4]

    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    ss_qss = ss_qss[filter_inds]

    return ss_qss

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
