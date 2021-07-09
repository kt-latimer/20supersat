"""
distribution of \sum\limits_{bin i}r_i*n_i in WCUs
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_vs_t, \
                              get_full_spectrum_dict, \
                              get_ss_qss_components
from phys_consts import *

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]

lwc_filter_val = 1.e-4
w_cutoff = 1

rmax = 102.e-6

change_CAS_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    sum_of_radii = get_sum_of_radii()
    make_hist(sum_of_radii)
    
def get_sum_of_radii():

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']
    A, B, meanr, nconc = get_ss_qss_components(ADLR_dict, full_spectrum_dict, \
            change_CAS_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    meanr = meanr[filter_inds]
    nconc = nconc[filter_inds]
    
    return meanr*nconc

def make_hist(sum_of_radii):

    fig, ax = plt.subplots()
    ax.hist(sum_of_radii, bins=30, density=False)

    ax.set_xlabel(r'$\sum_i r_i\cdot n_i$ ($\mu$m cm$^{-3}$)')
    ax.set_ylabel(r'Count')

    fig.suptitle('Distribution of sum of radii')

    outfile = FIG_DIR + 'sum_of_radii_hist_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
