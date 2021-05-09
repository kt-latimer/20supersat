"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np

from halo import CAS_bins, CIP_bins
from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[2]
magma_pink = colors_arr[5]
magma_orange = colors_arr[8]
                            
WRF_bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
WRF_bin_radii = WRF_bin_diams/2.

def main():

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [7, 1]})

    ax1.spines['right'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.tick_params(which='major', width=1.00)
    ax1.tick_params(which='major', length=5)
    ax1.tick_params(which='minor', width=0.75)
    ax1.tick_params(which='minor', length=2.5)
    ax1.patch.set_alpha(0.0)
    ax1.scatter(WRF_bin_radii*1.e6, [1 for r in WRF_bin_radii], \
                           c='b', s=6)
    for i, lo_bin_edge in enumerate(CAS_bins['lower']):
        up_bin_edge = CAS_bins['upper'][i]
        ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                        [4, 4], [5, 5], color='b', alpha=0.5)

    for i, lo_bin_edge in enumerate(CIP_bins['lower']):
        up_bin_edge = CIP_bins['upper'][i]
        ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                        [2, 2], [3, 3], color='r', alpha=0.5)

    ax1.set_xscale('log')
    ax0.set_xlabel(r'r ($\mu$m)')
    ax1.set_ylabel('Bins')
    ax1.set_ylim([0.5, 5.5])
    plt.tight_layout()

    outfile = FIG_DIR + 'test_axes_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
