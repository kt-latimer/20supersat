"""
make and save histograms showing SS_QSS distribution from HALO CDP measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CDP_bins, CIP_bins
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cdp_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

CDP_bin_radii = (CDP_bins['upper'] + CDP_bins['lower'])/4.
CIP_bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates[:1]:
        make_dsd_figs_for_date(date)

def make_dsd_figs_for_date(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdp_dict = np.load(cdpfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 273), \
                            (w > 1)))

    cdp_t = cdp_dict['data']['time'][filter_inds]

    i = 0
    while i < np.shape(cdp_t)[0]:
        cdp_dsd, cip_dsd = get_dsds(cdp_dict, cip_dict, filter_inds, i)
        make_dsd_fig(cdp_dsd, cip_dsd, date, i)
        i += 1 

def get_dsds(cdp_dict, cip_dict, filter_inds, i):

    cdp_dsd = []
    cip_dsd = []

    for j in range(8, 16):
        var_name = 'nconc_' + str(j)
        nconc_j = cdp_dict['data'][var_name][filter_inds][i]
        cdp_dsd.append(nconc_j)

    for j in range(1, 20):
        var_name = 'nconc_' + str(j)
        nconc_j = cip_dict['data'][var_name][filter_inds][i]
        cip_dsd.append(nconc_j)

    return cdp_dsd, cip_dsd

def make_dsd_fig(cdp_dsd, cip_dsd, date, i):

    fig, ax = plt.subplots()

    ax.plot(CDP_bin_radii[7:], cdp_dsd)
    ax.plot(CIP_bin_radii, cip_dsd)

    ax.set_xscale('log')
        
    outfile = FIG_DIR + 'cdp_compare_dsds_' + date + '_' \
                                + str(i) + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
