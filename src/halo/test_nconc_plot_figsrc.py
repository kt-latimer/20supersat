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
from halo.ss_functions import get_nconc_contribution_from_cas_var, \
                                get_meanr_contribution_from_cas_var
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]
magma_orange = colors_arr[1]

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = True 

CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    alldates_nconc = np.array([])

    for date in good_dates:
        nconc = get_test_nconc(date)
        alldates_nconc = np.concatenate((alldates_nconc, nconc))
        make_test_nconc_plot(nconc, date)

    make_test_nconc_plot(alldates_nconc, 'alldates')

def get_test_nconc(date):

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

    var_name = 'nconc_8'
    if change_cas_corr:
        var_name += '_corr'
    nconc = get_nconc_contribution_from_cas_var(var_name, adlr_dict, \
                cas_dict, change_cas_corr, cutoff_bins, incl_rain, incl_vent)

    return nconc[filter_inds]

def make_test_nconc_plot(nconc, date):

    fig, ax = plt.subplots()

    ax.hist(nconc, bins=30, density=True)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'test_nconc_plot_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
