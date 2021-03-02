"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_from_cas, get_lwc_from_cas_and_cip

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    lwc_cas_alldates = None
    lwc_cas_and_cip_alldates = None

    for date in good_dates:
        lwc_cas, lwc_cas_and_cip = get_lwc_data(date)
        lwc_cas_alldates = add_to_alldates_array(lwc_cas, lwc_cas_alldates)
        lwc_cas_and_cip_alldates = add_to_alldates_array(lwc_cas_and_cip, \
                                                    lwc_cas_and_cip_alldates)

    make_and_save_lwc_scatter(lwc_cas_alldates, lwc_cas_and_cip_alldates)

def add_to_alldates_array(lwc, lwc_alldates):

    if lwc_alldates is None:
        return lwc
    else:
        return np.concatenate((lwc_alldates, lwc))

def get_lwc_data(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    lwc_cas = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
    lwc_cas_and_cip = get_lwc_from_cas_and_cip(adlr_dict, cas_dict, cip_dict, \
                                                change_cas_corr, cutoff_bins)

    return lwc_cas, lwc_cas_and_cip 

def make_and_save_lwc_scatter(lwc_cas, lwc_cas_and_cip):

    fig, ax = plt.subplots()

    ax.scatter(lwc_cas, lwc_cas_and_cip)
    ax.set_xlabel('LWC from CAS (kg/kg)')
    ax.set_ylabel('LWC from CAS and CIP (kg/kg)')

    fig.suptitle('LWC values - HALO')

    outfile = FIG_DIR + 'compare_lwcs_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    
if __name__ == "__main__":
    main()
