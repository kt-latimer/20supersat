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
from revmywrf.ss_qss_calculations import get_lwc

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

HALO_DATA_DIR = '/global/home/users/kalatimer/proj/20supersat/data/revhalo/'
CAIPEEX_DATA_DIR = \
    '/global/home/users/kalatimer/proj/20supersat/data/revcaipeex/'

def main():

    halo_dict = np.load(HALO_DATA_DIR + 'v24_w_hist_cas_data_alldates.npy', \
                        allow_pickle=True).item()
    w_halo = halo_dict['w']
    caipeex_dict = np.load(CAIPEEX_DATA_DIR + 'v10_w_hist_cas_data_alldates.npy', \
                            allow_pickle=True).item()
    w_caipeex = caipeex_dict['w']
    
    versionnum = int(os.environ['versionnum'])
    (dummy_bool, cutoff_bins, full_ss, \
        incl_rain, incl_vent) = get_boolean_params(versionnum)
    versionstr = 'v' + str(versionnum) + '_'

    #laziest code of my life
    case_dir_name = 'C_BG/'

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

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    w_unpolluted = w[filter_inds]
    
    del lwc, temp, w, filter_inds #for memory

    case_dir_name = 'C_PI/'

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

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    w_polluted = w[filter_inds]
    
    del lwc, temp, w, filter_inds #for memory
    
    make_and_save_w_hist(w_polluted, w_unpolluted, w_halo, \
                            w_caipeex, cutoff_bins, full_ss, \
                            incl_rain, incl_vent, versionstr)

def make_and_save_w_hist(w_polluted, w_unpolluted, w_halo, \
                            w_caipeex, cutoff_bins, full_ss, \
                            incl_rain, incl_vent, versionstr):

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    n_wrf, bins_wrf, patches_wrf = ax.hist(w_polluted, bins=30, \
            density=True, label='WRF - polluted', \
            facecolor=(0, 0, 1, 0.0), edgecolor='blue', \
            histtype='stepfilled', linewidth=2)
    ax.hist(w_unpolluted, bins=bins_wrf, density=True, \
            label='WRF - unpolluted', facecolor=(1, 0, 0, 0.0), \
            edgecolor='red', histtype='stepfilled', linewidth=2) 
    ax.hist(w_halo, bins=bins_wrf, density=True, label='HALO', \
            facecolor=(0, 1, 0, 0.0), edgecolor='green', \
            histtype='stepfilled', linewidth=2)
    ax.hist(w_caipeex, bins=bins_wrf, density=True, label='CAIPEEX', \
            facecolor=(0, 0, 0, 0.0), edgecolor='black', \
            histtype='stepfilled', linewidth=2)
    ax.set_xlabel('w (m/s)')
    ax.set_ylabel('Normalized count')
    ax.set_title('Combined vert wind vel distb' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent) \
                    + ', full_ss=' + str(full_ss))
    plt.legend()

    outfile = FIG_DIR + versionstr + 'combined_w_hist_figure_2.png'
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
