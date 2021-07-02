"""
Make plots comparing d(N*r^3)/dlogDp for HALO and both WRF simulations
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.ss_functions import get_nconc_contribution_from_spectrum_var, \
                                get_meanr_contribution_from_spectrum_var, \
                                get_full_spectrum_bin_radii, \
                                get_full_spectrum_bin_dlogDp, \
                                get_full_spectrum_dict, get_lwc_vs_t
from halo.utils import linregress
from wrf import DATA_DIR as WRF_DATA_DIR, WRF_bin_radii, WRF_bin_bin_dlogDp

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'halo': colors_arr[2], 'wrf_poll': colors_arr[5], \
                                    'wrf_unpoll': colors_arr[8]}
                            
lwc_cutoff_val = 1.e-4
w_cutoff_val = 1

rmax = 102.e-6

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
case_color_key_dict = {'Polluted': 'wrf_poll', 'Unpolluted': 'wrf_unpoll'}

change_CAS_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = False 

HALO_bin_radii = get_full_spectrum_bin_radii(CAS_bins, CIP_bins, 'log')
HALO_bin_dlogDp = get_full_spectrum_bin_dlogDp(CAS_bins, CIP_bins)

def main():

    spectrum_dict = get_spectrum_dict() 
    make_vol_distribution_fig(spectrum_dict)

def get_spectrum_dict():

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']
    z = ADLR_dict['data']['alt']

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (z > 1500), \
                            (z < 2500), \
                            (w > w_cutoff_val)))

    meanr_spectrum_dict = {}
    for var_name in spectrum_dict['data'].keys():
        meanr = get_meanr_contribution_from_spectrum_var(var_name, ADLR_dict, \
            spectrum_dict, cutoff_bins, incl_rain, incl_vent, HALO_bin_radii)
        meanr = meanr[filter_inds]
        meanr_var_name = var_name.replace('nconc', 'meanr')
        meanr_spectrum_dict[meanr_var_name] = meanr

    return meanr_spectrum_dict

def make_vol_distribution_fig(spectrum_dict):

    fig, ax = plt.subplots()

    HALO_meanr = np.array([np.nanmean(spectrum_dict[key]) \
                            for key in spectrum_dict.keys()])
    
    if incl_vent:
        vent_key = 'with_vent'
    else:
        vent_key = 'no_vent'

    for case_label in case_label_dict.keys():
        WRF_dict = get_WRF_dict(case_label)
        WRF_nconc = WRF_dict['data'][vent_key]
        ax.plot(WRF_bin_radii*1.e6,
                WRF_nconc*WRF_bin_radii**3.*1.e12/WRF_bin_dlogDp, \
                color=colors_dict[case_color_key_dict[case_label]], \
                linestyle=linestyles_dict[versionstr], \
                label='WRF ' + case_label)
    ax.plot(HALO_bin_radii*1.e6, \
            HALO_meanr*HALO_bin_radii**2.*1.e12/HALO_bin_dlogDp, \
            color=colors_dict['halo'], label='HALO')

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(y_labels_dict[vent_key])

    ax.set_xscale('log')

    ax.legend()
    ax.set_title('Drop volume distributions - 1.5-2.5 km')
    
    outfile = FIG_DIR + 'FINAL_compare_vol_distributions_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_WRF_dict(case_label):

    #misleading file name but each bin is avg of n(r) for WCU in 1.5-2.5
    #km altitude slice
    WRF_filename = WRF_DATA_DIR + 'avg_dsd_for_slice_' + case_label + '_data.npy'
    WRF_dict = np.load(WRF_filename, allow_pickle=True).item()

    return WRF_dict

if __name__ == "__main__":
    main()
