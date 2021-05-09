"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np

from halo import CAS_bins, CIP_bins, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[2]
magma_pink = colors_arr[5]
magma_orange = colors_arr[8]

def main():

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

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
    for i, lo_bin_edge in enumerate(CAS_bins['lower']):
        up_bin_edge = CAS_bins['upper'][i]/2.
        lo_bin_edge = lo_bin_edge/2.
        if i == 0:
            ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                [4, 4], [5, 5], color=magma_red, alpha=0.5, label='CAS')
        else:
            ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                        [4, 4], [5, 5], color=magma_red, alpha=0.5)

    for i, lo_bin_edge in enumerate(CIP_bins['lower']):
        up_bin_edge = CIP_bins['upper'][i]/2.
        lo_bin_edge = lo_bin_edge/2.
        if i == 0:
            ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                [2, 2], [3, 3], color=magma_orange, alpha=0.5, label='CIP')
        else:
            ax1.fill_between([lo_bin_edge*1.e6, up_bin_edge*1.e6], \
                    [2, 2], [3, 3], color=magma_orange, alpha=0.5)

    ax1.set_xscale('log')
    ax0.set_xlabel(r'r ($\mu$m)')
    ax1.set_xlabel(r'r ($\mu$m)')
    ax1.set_ylabel('Bins')
    ax1.set_ylim([0.5, 5.5])
    ax1.legend()
    plt.tight_layout()

    outfile = FIG_DIR + 'test_axes_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
