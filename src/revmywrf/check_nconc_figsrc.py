"""
check my nconc calculations against wrf outputs
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_nconc

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = False 
incl_rain = False
incl_vent = False

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_nconc_comparison_fig(case_label, \
                                            case_label_dict[case_label])

def make_and_save_nconc_comparison_fig(case_label, case_dir_name):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    nconc_wrf = met_vars['nconc_cloud'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    #get dsd sum file variables for 25um radius cutoff for cloud
    dsdsum_file_25um_cutoff = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars_25um_cutoff = dsdsum_file_25um_cutoff.variables
 
    nconc_25um_cutoff = get_nconc(dsdsum_vars_25um_cutoff, \
                                    cutoff_bins, incl_rain, incl_vent)

    lwc_25um_cutoff = get_lwc(met_vars, dsdsum_vars_25um_cutoff, \
                                cutoff_bins, incl_rain, incl_vent)

    dsdsum_file_25um_cutoff.close()

    #get dsd sum file variables for 51um radius cutoff for cloud
    dsdsum_file_51um_cutoff = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars_51um_cutoff = dsdsum_file_51um_cutoff.variables
 
    nconc_51um_cutoff = get_nconc(dsdsum_vars_51um_cutoff, \
                                    cutoff_bins, incl_rain, incl_vent)

    dsdsum_file_51um_cutoff.close()

    #close files for memory
    met_file.close()

    filter_inds = np.logical_and.reduce((
                    (lwc_25um_cutoff > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    nconc_wrf = nconc_wrf[filter_inds]
    nconc_25um_cutoff = nconc_25um_cutoff[filter_inds]
    nconc_51um_cutoff = nconc_51um_cutoff[filter_inds]

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(24, 12)
    ax[0].scatter(nconc_wrf, nconc_25um_cutoff)
    ax[0].set_xlabel('nconc cloud from WRF (m^-3)')
    ax[0].set_ylabel('nconc cloud from me, 25um radius cutoff (m^-3)')
    ax[1].scatter(nconc_wrf, nconc_51um_cutoff)
    ax[1].set_xlabel('nconc cloud from WRF (m^-3)')
    ax[1].set_ylabel('nconc cloud from me, 51um radius cutoff (m^-3)')
    outfile = FIG_DIR + versionstr + 'check_nconc_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
