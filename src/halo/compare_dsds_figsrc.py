"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

CAS_bin_radii = (CAS_bins['upper'] + CAS_bins['lower'])/4.
CIP_bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates[:1]:
        make_dsd_figs_for_date(date)

def make_dsd_figs_for_date(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 273), \
                            (w > 1)))

    cas_t = cas_dict['data']['time'][filter_inds]

    i = 0
    while i < np.shape(cas_t)[0]:
        cas_dsd, cip_dsd = get_dsds(cas_dict, cip_dict, filter_inds, i)
        make_dsd_fig(cas_dsd, cip_dsd, date, i)
        i += 1 

def get_dsds(cas_dict, cip_dict, filter_inds, i):

    cas_dsd = []
    cip_dsd = []

    xi = cas_dict['data']['xi'][filter_inds]
    PAS = cas_dict['data']['PAS'][filter_inds]
    TAS = cas_dict['data']['TAS'][filter_inds]

    volume_corr_factor = xi/(PAS/TAS)

    for j in range(5, 17):
        var_name = 'nconc_' + str(j)
        nconc_j = cas_dict['data'][var_name][filter_inds][i]
        cas_dsd.append(nconc_j*volume_corr_factor[i])

    for j in range(1, 20):
        var_name = 'nconc_' + str(j)
        nconc_j = cip_dict['data'][var_name][filter_inds][i]
        cip_dsd.append(nconc_j)

    return np.array(cas_dsd), np.array(cip_dsd)

def make_dsd_fig(cas_dsd, cip_dsd, date, i):

    fig, ax = plt.subplots()

    ax.scatter(CAS_bin_radii*1.e6, cas_dsd*1.e-6)
    ax.scatter(CIP_bin_radii*1.e6, cip_dsd*1.e-6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1.e-10, 1.e3])
        
    outfile = FIG_DIR + 'compare_dsds_' + date + '_' + str(i) + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
