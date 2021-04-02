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

CDP_bin_radii = (CDP_bins['upper'] + CDP_bins['lower'])/4.
CIP_bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

CIP_dlogDp = np.log10(CIP_bins['upper']/CIP_bins['lower'])
CDP_dlogDp = np.log10(CDP_bins['upper']/CDP_bins['lower'])

start_times = [56141, 56795, 57379]
end_times = [56719, 57286, 58062]

def main():
    
    good_dates = ['20140921']

    for date in good_dates:
        make_dsd_figs_for_date(date)

def make_dsd_figs_for_date(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdp_dict = np.load(cdpfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    cdp_t = cdp_dict['data']['time']

    start_inds, end_inds = get_start_end_inds(cdp_t, start_times, end_times)
    print(start_inds, end_inds)

    for i, start_ind in enumerate(start_inds):
        end_ind = end_inds[i]
        cdp_dsd, cip_dsd = get_dsds(cdp_dict, cip_dict, start_ind, end_ind)
        make_dsd_fig(cdp_dsd, cip_dsd, start_ind)

def get_start_end_inds(cdp_t, start_times, end_times):

    start_inds = []
    end_inds = []

    for i, t in enumerate(cdp_t):
        if t in start_times:
            start_inds.append(i)
        elif t in end_times:
            end_inds.append(i)

    return start_inds, end_inds
    
def get_dsds(cdp_dict, cip_dict, start_ind, end_ind):

    cdp_dsd = []
    cip_dsd = []
    tot = 0

    for j in range(1, 16):
        var_name = 'nconc_' + str(j)
        nconc_j = np.nanmean(cdp_dict['data'][var_name][start_ind:end_ind] \
                                                        /CDP_dlogDp[j - 1])
        cdp_dsd.append(nconc_j)

        if j >= 9:
            tot += np.nanmean(cdp_dict['data'][var_name][start_ind:end_ind])
            print(j, tot)

    print('cdp: ', tot)

    for j in range(1, 20):
        var_name = 'nconc_' + str(j)
        nconc_j = np.nanmean(cip_dict['data'][var_name][start_ind:end_ind] \
                                                        /CIP_dlogDp[j - 1])
        cip_dsd.append(nconc_j)

    tot = np.nanmean(cip_dict['data']['nconc_1'][start_ind:end_ind])
    print('cip: ', tot) 

    return np.array(cdp_dsd), np.array(cip_dsd)

def make_dsd_fig(cdp_dsd, cip_dsd, start_ind):

    fig, ax = plt.subplots()

    ax.scatter(CDP_bin_radii*1.e6, cdp_dsd*1.e-6)
    ax.scatter(CIP_bin_radii*1.e6, cip_dsd*1.e-6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1.e-6, 1.e3])
        
    outfile = FIG_DIR + 'mei_compare_dsds_' + str(start_ind) + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
