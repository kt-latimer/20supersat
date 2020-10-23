"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_meanr, get_nconc

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
    versionstr = 'v' + str(versionnum) + '_log_'

    for case_label in case_label_dict.keys():
        print(case_label)
        search_for_and_print_outliers(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)
        print()

def search_for_and_print_outliers(case_label, case_dir_name, \
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
    meanr = get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    if case_label == 'Polluted':
        filter_inds1 = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (meanr < 0), \
                        (temp > 273)))
        filter_inds2 = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (meanr > 0.0002), \
                        (temp > 273)))
        print_point_info(filter_inds1, lwc, meanr, nconc, pres, temp, w, z)
        print_point_info(filter_inds2, lwc, meanr, nconc, pres, temp, w, z)
    else:
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (meanr > 0.1), \
                        (temp > 273)))
        print_point_info(filter_inds, lwc, meanr, nconc, pres, temp, w, z)
    
    del lwc, meanr, nconc, pres, temp, w, z #for memory

def print_point_info(filter_inds, lwc, meanr, nconc, pres, temp, w, z):

    print('lwc: ', lwc[filter_inds])
    print('meanr: ', meanr[filter_inds])
    print('nconc: ', nconc[filter_inds])
    print('pres: ', pres[filter_inds])
    print('temp: ', temp[filter_inds])
    print('w: ', w[filter_inds])
    print('z: ', z[filter_inds])

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
